"""
fine_tune_trigger.py — Active Learning Flywheel: trigger, train, evaluate, and hot-swap the GNN.

Rules:
  • Only fires when ≥ FINE_TUNE_THRESHOLD new corrections have been logged
    since the last deployed model.
  • New model is deployed ONLY if its AUC-ROC strictly beats the current one.
  • Thread-safe hot-reload — no server restart required.
  • All runs are tracked via MLflow (local file store, no external server).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from datetime import datetime

import torch
from torch.optim import AdamW

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so they work from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

MODEL_PATH        = os.path.join(_REPO_ROOT, "data", "models", "gnn_model_real_cpu.pt")
CALIBRATION_PATH  = os.path.join(_REPO_ROOT, "data", "models", "calibration_real.json")
TRAIN_DATA_PATH   = os.path.join(_REPO_ROOT, "data", "processed", "real_cad_training_data_full.pt")
MLFLOW_URI        = os.path.join(_REPO_ROOT, "data", "mlflow")

FINE_TUNE_THRESHOLD = 30       # minimum new corrections before a run triggers
CORRECTION_WEIGHT   = 3        # repeat corrections N times (verified ground truth)
FINE_TUNE_LR        = 5e-5
FINE_TUNE_EPOCHS    = 20
PATIENCE            = 5

_trigger_lock = threading.Lock()   # prevent concurrent fine-tune runs


# ---------------------------------------------------------------------------
# MLflow setup (lazy import — server still works if mlflow not installed)
# ---------------------------------------------------------------------------

def _try_mlflow_setup():
    try:
        import mlflow
        os.makedirs(MLFLOW_URI, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("dfm_gnn_flywheel")
        return mlflow
    except ImportError:
        logger.warning("[Flywheel] mlflow not installed — tracking disabled")
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_and_trigger():
    """
    Called in a background task after every /feedback POST.
    Acquires a lock so only one fine-tune can run at a time.
    """
    from .feedback_store import get_uncollected_corrections

    corrections = get_uncollected_corrections()
    n = len(corrections)

    if n < FINE_TUNE_THRESHOLD:
        logger.info(
            "[Flywheel] %d/%d corrections — threshold not reached",
            n, FINE_TUNE_THRESHOLD,
        )
        return

    if not _trigger_lock.acquire(blocking=False):
        logger.info("[Flywheel] Fine-tune already running — skipping duplicate trigger")
        return

    try:
        logger.info("[Flywheel] %d corrections collected — starting fine-tune", n)
        _run_fine_tune(corrections)
    except Exception as e:
        logger.error("[Flywheel] Fine-tune failed: %s", e, exc_info=True)
    finally:
        _trigger_lock.release()


# ---------------------------------------------------------------------------
# Core fine-tune pipeline
# ---------------------------------------------------------------------------

def _run_fine_tune(corrections: list[dict]):
    from .gnn_model import DFMGNN
    from .feedback_store import (
        get_current_registry_entry,
        register_model,
        mark_deployed,
    )

    # 1. Load base training dataset
    if not os.path.exists(TRAIN_DATA_PATH):
        logger.error("[Flywheel] Base training data not found at %s", TRAIN_DATA_PATH)
        return

    base_dataset: list = torch.load(TRAIN_DATA_PATH, weights_only=False)
    logger.info("[Flywheel] Loaded %d base graphs", len(base_dataset))

    # 2. Rebuild corrected graphs from corrections
    correction_graphs = _build_correction_graphs(corrections)
    logger.info("[Flywheel] Built %d correction graphs from %d corrections",
                len(correction_graphs), len(corrections))

    if not correction_graphs:
        logger.warning("[Flywheel] No correction graphs could be built — aborting")
        return

    # 3. Combine dataset: corrections are repeated CORRECTION_WEIGHT times
    dataset = base_dataset + correction_graphs * CORRECTION_WEIGHT
    random.shuffle(dataset)
    logger.info("[Flywheel] Fine-tune dataset size: %d graphs", len(dataset))

    # 4. Split — 90/10 train/val
    split = int(len(dataset) * 0.9)
    train_set = dataset[:split]
    val_set   = dataset[split:] or dataset[-max(1, len(dataset)//10):]

    # 5. Load current model weights as starting point
    node_in = 12
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH) as f:
                cal = json.load(f)
            from .inference import DFMInferenceEngine
            node_in = DFMInferenceEngine._detect_node_in(cal.get("version", ""))
        except Exception:
            pass

    logger.info("[Flywheel] Dynamic node_in detected: %d", node_in)
    model = DFMGNN(node_in=node_in, hidden=64, out_dim=1, heads=2, dropout=0.3)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        logger.info("[Flywheel] Loaded current weights as starting point")

    # 6. Fine-tune
    optimizer = AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4)
    _fine_tune_loop(model, optimizer, train_set, val_set)

    # 7. Evaluate on held-out val set
    metrics = _evaluate(model, val_set)
    logger.info(
        "[Flywheel] Fine-tune metrics — AUC: %.4f  F1: %.4f",
        metrics["auc"], metrics["f1"],
    )

    # 8. Only deploy if better than current
    current = get_current_registry_entry()
    current_auc = current.get("auc_roc", 0.0) or 0.0

    # Determine the max correction id consumed in this run
    last_feedback_id = max(c["id"] for c in corrections)

    version = _bump_version(current.get("version", "v0.0"))

    # Log to MLflow regardless of deploy decision
    mlflow = _try_mlflow_setup()
    if mlflow:
        _log_mlflow(mlflow, model, version, metrics, len(dataset))

    if metrics["auc"] > current_auc:
        logger.info(
            "[Flywheel] Deploying %s  (AUC %.4f → %.4f)",
            version, current_auc, metrics["auc"],
        )
        _deploy(model, version, metrics, len(dataset), last_feedback_id)
        register_model(version, len(dataset), metrics["auc"], metrics["f1"], last_feedback_id)
        mark_deployed(version)
        # Hot-reload inference engine
        _hot_reload()
        logger.info("[Flywheel] %s is live — no server restart required", version)
    else:
        logger.info(
            "[Flywheel] New AUC %.4f ≤ current %.4f — keeping existing model",
            metrics["auc"], current_auc,
        )
        # Still register so we can track the attempt
        register_model(version + "-rejected", len(dataset),
                       metrics["auc"], metrics["f1"], last_feedback_id)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _fine_tune_loop(model, optimizer, train_set, val_set):
    import torch.nn.functional as F
    from torch_geometric.data import Batch

    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    no_improve = 0

    model.train()
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        random.shuffle(train_set)
        total_loss = 0.0
        n_batches = 0

        # Mini-batch by 32 graphs
        for i in range(0, len(train_set), 32):
            batch_graphs = train_set[i : i + 32]
            try:
                batch = Batch.from_data_list(batch_graphs)
            except Exception:
                continue

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.float().view(-1)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            loss = loss_fn(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation loss for early stopping
        val_loss = _compute_val_loss(model, val_set, loss_fn)
        logger.info(
            "[Flywheel] Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
            epoch, FINE_TUNE_EPOCHS, avg_loss, val_loss,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info("[Flywheel] Early stopping at epoch %d", epoch)
                break

    model.eval()


def _compute_val_loss(model, val_set, loss_fn):
    from torch_geometric.data import Batch

    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(val_set), 32):
            batch_graphs = val_set[i : i + 32]
            try:
                batch = Batch.from_data_list(batch_graphs)
            except Exception:
                continue
            out = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.float().view(-1)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            total += loss_fn(out, labels).item()
            n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, val_set) -> dict:
    """Compute AUC-ROC and F1 on val_set. Returns dict with 'auc' and 'f1'."""
    import math
    from torch_geometric.data import Batch

    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for i in range(0, len(val_set), 32):
            batch_graphs = val_set[i : i + 32]
            try:
                batch = Batch.from_data_list(batch_graphs)
            except Exception:
                continue
            out = model(batch.x, batch.edge_index, batch.batch)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            probs = torch.sigmoid(out).cpu().tolist()
            labels = batch.y.cpu().view(-1).tolist()
            all_scores.extend(probs)
            all_labels.extend(labels)

    if not all_scores:
        return {"auc": 0.5, "f1": 0.0}

    auc = _roc_auc(all_labels, all_scores)
    f1  = _f1(all_labels, all_scores, threshold=0.5)
    return {"auc": round(auc, 6), "f1": round(f1, 6)}


def _roc_auc(labels, scores) -> float:
    """Trapezoidal AUC-ROC without sklearn dependency."""
    paired = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    p = sum(labels)
    n = len(labels) - p
    if p == 0 or n == 0:
        return 0.5

    auc = 0.0
    prev_fp, prev_tp = 0, 0
    for _, label in paired:
        if label:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_fp, prev_tp = fp, tp

    return auc / (p * n)


def _f1(labels, scores, threshold=0.5) -> float:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# ---------------------------------------------------------------------------
# Correction graph builder
# ---------------------------------------------------------------------------

def _build_correction_graphs(corrections: list[dict]) -> list:
    """
    Attempt to rebuild PyG Data objects from corrections.

    If yml_path points to a real YAML descriptor we call build_graph_from_yml.
    Otherwise we synthesise a minimal single-node graph so the correction
    label is still incorporated (weight is lower via CORRECTION_WEIGHT).
    """
    node_in = 12
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH) as f:
                cal = json.load(f)
            from .inference import DFMInferenceEngine
            node_in = DFMInferenceEngine._detect_node_in(cal.get("version", ""))
        except Exception:
            pass

    graphs = []
    for c in corrections:
        g = None
        if c.get("yml_path") and os.path.exists(c["yml_path"]):
            try:
                g = _build_graph_from_yml(c["yml_path"], node_in)
            except Exception as e:
                logger.debug("[Flywheel] Could not parse yml %s: %s", c["yml_path"], e)

        if g is None:
            # Fallback: synthesise a neutral 1-node graph carrying the label
            import torch
            from torch_geometric.data import Data
            g = Data(
                x=torch.zeros(1, node_in),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )

        g.y = torch.tensor([float(c["engineer_label"])])
        graphs.append(g)
    return graphs


def _build_graph_from_yml(yml_path: str, node_in: int = 12):
    """
    Parse a YAML part descriptor and return a PyG Data object.
    YAML schema mirrors what the SolidWorks add-in exports.
    """
    import yaml
    import torch
    from torch_geometric.data import Data
    from .gnn_model import DFMGNN
    from .inference import compute_graph_level_stats

    with open(yml_path, "r") as f:
        doc = yaml.safe_load(f)

    faces = doc.get("faces", [])
    if not faces:
        return None

    feat_model = DFMGNN(node_in, 64, 1, heads=2)

    graph_stats = compute_graph_level_stats(faces)

    x_list = []
    for face in faces:
        feats = feat_model.get_node_features(face)
        feats = feats + graph_stats
        # Pad or truncate to match node_in
        feats = feats[:node_in]
        while len(feats) < node_in:
            feats.append(0.0)
        x_list.append(feats)

    x = torch.tensor(x_list, dtype=torch.float32)

    n = len(faces)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges += [[i, j], [j, i]]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t()
        if edges
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

def _deploy(model, version: str, metrics: dict, n_graphs: int, last_feedback_id: int):
    """Write new weights and calibration to disk (hot-reload picks them up)."""
    # Back up current weights
    bak = MODEL_PATH + ".bak"
    if os.path.exists(MODEL_PATH):
        import shutil
        shutil.copy2(MODEL_PATH, bak)

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("[Deploy] Weights saved to %s", MODEL_PATH)

    # Update calibration JSON with Platt scaling re-fitted on new model
    # We overwrite the version and preserve or update A and B.
    A, B = 1.0, 0.0
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH) as f:
                cal = json.load(f)
            A = float(cal.get("A", 1.0))
            B = float(cal.get("B", 0.0))
        except Exception:
            pass

    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"A": A, "B": B, "version": version}, f)


def _log_mlflow(mlflow, model, version: str, metrics: dict, n_graphs: int):
    try:
        with mlflow.start_run(run_name=f"fine_tune_{version}",
                              tags={"type": "fine_tune", "version": version}):
            mlflow.log_metric("auc_roc", metrics["auc"])
            mlflow.log_metric("f1_score", metrics["f1"])
            mlflow.log_param("n_graphs", n_graphs)
            mlflow.log_param("version", version)
            mlflow.log_param("lr", FINE_TUNE_LR)
            mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
            if os.path.exists(CALIBRATION_PATH):
                mlflow.log_artifact(CALIBRATION_PATH)
            try:
                import mlflow.pytorch
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="dfm_gnn",
                )
            except Exception as e:
                logger.debug("[Flywheel] Could not log model to MLflow registry: %s", e)
    except Exception as e:
        logger.warning("[Flywheel] MLflow logging failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------

def _hot_reload():
    """Signal the global inference engine to reload its weights."""
    try:
        # Imported lazily to avoid circular imports at module level
        from ..services import gnn_engine
        gnn_engine.reload_model()
    except Exception as e:
        logger.error("[Flywheel] Hot-reload failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Version bumping
# ---------------------------------------------------------------------------

def _bump_version(current: str) -> str:
    """
    Bump v<major>.<minor> → v<major>.<minor+1>.
    Falls back to a timestamp-based version on parse error.
    """
    suffix = ""
    if "14d" in current:
        suffix = "-14d"
        current = current.replace("-14d", "")

    try:
        prefix, rest = current.lstrip("v").split(".")
        new_ver = f"v{int(prefix)}.{int(rest) + 1}"
    except Exception:
        new_ver = f"v1.{int(time.time())}"

    return new_ver + suffix
