"""
inference.py — Thread-safe DFM GNN inference engine with:
  • Hot-reload (no server restart on fine-tune deploy)
  • Model version tracking from calibration JSON
  • MLflow prediction logging (local file store, no external server)
"""

import json
import logging
import math
import os
import threading

import torch

from ..core.models import PartMetadata

logger = logging.getLogger(__name__)

# Hybrid XGBoost support (optional)
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    import numpy as np
    XGB_AVAILABLE = True
except ImportError:
    pass

ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

from .gnn_model import DFMGNN


# Legacy 2-layer GATv2 architecture for loading old v1.x weights
class _Legacy2LayerDFMGNN(torch.nn.Module):
    """Backward-compatible 2-layer GATv2 that matches v1.x checkpoints.
    Architecture: conv1(in->64) -> conv2(64->64) -> mean+max pool(128)
                  -> lin1(128->64) -> bn3(64) -> lin2(64->32) -> lin3(32->1)
    """
    def __init__(self, node_in=14, hidden=64, out_dim=1, heads=2, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        import torch.nn as nn
        # Old architecture: out_channels=32 per head, concat=True → 64-D output
        per_head = hidden // heads  # 32
        self.conv1 = GATv2Conv(node_in, per_head, heads=heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = GATv2Conv(hidden, per_head, heads=heads, concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.lin1 = nn.Linear(hidden * 2, hidden)   # mean+max pool concat
        self.bn3 = nn.BatchNorm1d(hidden)
        self.lin2 = nn.Linear(hidden, 32)
        self.lin3 = nn.Linear(32, out_dim)
        self.dropout = nn.Dropout(dropout)
        self._node_in = node_in

    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool, global_max_pool
        import torch.nn.functional as F
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.leaky_relu(h, 0.01)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.leaky_relu(h, 0.01)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        g = torch.cat([mean_pool, max_pool], dim=1)
        g = F.leaky_relu(self.lin1(g), 0.01)
        g = self.bn3(g)
        g = self.dropout(g)
        g = F.leaky_relu(self.lin2(g), 0.01)
        return self.lin3(g).squeeze(-1)

    def get_node_features(self, face_data):
        """Same feature extraction as current DFMGNN."""
        return DFMGNN(self._node_in, 64, 1).get_node_features(face_data)

    def get_face_embeddings_and_scores(self, x, edge_index):
        import torch.nn.functional as F
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.leaky_relu(h, 0.01)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.leaky_relu(h, 0.01)
        # Per-face scores through MLP
        g = F.leaky_relu(self.lin1(torch.cat([h, h], dim=1)), 0.01)
        g = self.bn3(g)
        g = F.leaky_relu(self.lin2(g), 0.01)
        face_logits = self.lin3(g).squeeze(-1)
        face_probs = torch.sigmoid(face_logits)
        return h, face_probs

def compute_graph_level_stats(faces):
    """
    2 graph-level stats broadcast to every node:
      - min_plane_draft / 45  (normalised)
      - frac_plane            (fraction of faces that are Plane)
    Used by both 12-D and 14-D models.
    """
    plane_faces = []
    for f in faces:
        ftype = ""
        if hasattr(f, "face_type"):
            ftype = getattr(f.face_type, "value", str(f.face_type))
        elif isinstance(f, dict):
            val = f.get("face_type", "")
            ftype = getattr(val, "value", str(val))
        if ftype == "Plane":
            plane_faces.append(f)

    plane_drafts = []
    for f in plane_faces:
        da = None
        if hasattr(f, "draft_angle_deg"):
            da = f.draft_angle_deg
        elif isinstance(f, dict):
            da = f.get("draft_angle_deg")
        if da is not None:
            plane_drafts.append(da)

    min_plane_draft = min(plane_drafts) / 45.0 if plane_drafts else 1.0
    frac_plane = len(plane_faces) / len(faces) if faces else 0.0

    return [min_plane_draft, frac_plane]


# Backward-compatible alias
compute_14d_graph_stats = compute_graph_level_stats


class DFMInferenceEngine:
    def __init__(self, model_path: str | None = None,
                 threshold_path: str | None = None):
        self._lock = threading.RLock()
        self._model_path = model_path
        self._threshold_path = threshold_path

        # Public attributes — all mutated inside _lock during reload
        self.model: DFMGNN | None = None
        self.session = None           # ONNX session (optional)
        self.threshold: float = 0.5
        self.available: bool = False
        self.cal_A: float = 1.0
        self.cal_B: float = 0.0
        self.model_version: str = "v0.0"
        self.inference_mode: str = "gnn"  # "gnn" or "hybrid"

        # Hybrid XGBoost model (optional)
        self.xgb_model = None
        self.hybrid_cal_A: float = 1.0
        self.hybrid_cal_B: float = 0.0
        self.hybrid_embed_dim: int = 0

        # MLflow (optional — graceful if not installed)
        self._mlflow = self._init_mlflow()

        self.load_model()
        self._load_hybrid_model()

    # -----------------------------------------------------------------------
    # Load / Reload
    # -----------------------------------------------------------------------

    def load_model(self):
        """Initial load — called once at startup."""
        with self._lock:
            self._load_internal()

    def reload_model(self):
        """
        Thread-safe hot reload after a fine-tune deploy.
        Performs dummy verification outside the lock, then swaps atomically.
        """
        logger.info("[Inference] Hot-reloading model…")
        success = self._hot_swap()
        if not success:
            logger.error("[Inference] Hot-swap failed — keeping current model.")

    def _hot_swap(self, model_path=None, cal_path=None) -> bool:
        """
        Performs I/O, loading, and verification on the new GNN model and ONNX session
        OUTSIDE the lock. Returns True on success, False otherwise.
        If successful, swaps references atomically inside the lock.
        """
        path = model_path or self._model_path
        if not path or not os.path.exists(path):
            logger.warning("[Inference] Model path not found for hot-swap: %s", path)
            return False

        model_dir = os.path.dirname(path)
        cal_path = cal_path or os.path.join(model_dir, "calibration_real.json")
        
        node_in = 12  # default; overridden by calibration version
        cal_A = 1.0
        cal_B = 0.0
        version = self.model_version

        # 1. Load calibration file outside lock
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    cal = json.load(f)
                cal_A = float(cal.get("A", 1.0))
                cal_B = float(cal.get("B", 0.0))
                version = cal.get("version", version)
                node_in = self._detect_node_in(version)
            except Exception as e:
                logger.warning("[Inference] Could not read calibration for hot-swap: %s", e)
                return False

        # 2. Load and verify PyTorch model weights outside lock
        try:
            new_model = DFMGNN(node_in, 64, 1, heads=2, dropout=0.3)
            new_model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            new_model.eval()

            # Run dummy verification to prevent loading incorrect weights
            dummy_x = torch.randn(5, node_in, dtype=torch.float32)
            dummy_edge = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 4]], dtype=torch.long)
            dummy_batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
            with torch.no_grad():
                _ = new_model(dummy_x, dummy_edge, dummy_batch)
                _, _ = new_model.get_face_embeddings_and_scores(dummy_x, dummy_edge)
        except Exception as e:
            logger.error("[Inference] Hot-swap dummy PyTorch inference failed: %s", e)
            return False

        # 3. Load and verify ONNX session outside lock
        new_session = None
        if ONNX_AVAILABLE:
            onnx_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(".onnx") and ".stale" not in f
            ]
            if onnx_files:
                onnx_path = os.path.join(model_dir, onnx_files[0])
                try:
                    sess = ort.InferenceSession(onnx_path)
                    shape = sess.get_inputs()[0].shape
                    feat_dim = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else None
                    if feat_dim is not None and feat_dim != node_in:
                        logger.warning("[Inference] ONNX expects %d-D, but PyTorch uses %d-D. Hot-swap skipping ONNX.", feat_dim, node_in)
                    else:
                        # Test dummy verification on ONNX session
                        dummy_x_np = torch.randn(5, node_in, dtype=torch.float32).numpy()
                        dummy_edge_np = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 4]], dtype=torch.long).numpy()
                        dummy_batch_np = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long).numpy()
                        ort_inputs = {
                            "x": dummy_x_np,
                            "edge_index": dummy_edge_np,
                            "batch": dummy_batch_np
                        }
                        _ = sess.run(None, ort_inputs)
                        new_session = sess
                        logger.info("[Inference] ONNX dummy test OK")
                except Exception as e:
                    logger.warning("[Inference] ONNX load or test failed: %s", e)

        # 4. Atomic reference swap inside RLock
        with self._lock:
            self.model = new_model
            self.session = new_session
            self.cal_A = cal_A
            self.cal_B = cal_B
            self.model_version = version
            self.node_in = node_in
            self.available = True
            logger.info("[Inference] Hot-swap completed atomically. Model version: %s (in_channels=%d)", version, node_in)

        return True

    @staticmethod
    def _detect_node_in(version: str) -> int:
        """Detect feature dimensionality from calibration version string.
        '12d' → 12,  '14d' → 14,  default → 12.
        """
        v = (version or "").lower()
        if "14d" in v:
            return 14
        if "12d" in v:
            return 12
        return 12   # default for new models

    def _load_internal(self):
        """Must be called while holding self._lock."""
        path = self._model_path
        if not path or not os.path.exists(path):
            logger.warning("[Inference] Model path not found: %s", path)
            return

        # Determine node input dimension from calibration version first
        node_in = 12
        self.cal_A = 1.0
        self.cal_B = 0.0

        model_dir = os.path.dirname(path)
        cal_path = os.path.join(model_dir, "calibration_real.json")
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    cal = json.load(f)
                self.cal_A = float(cal.get("A", 1.0))
                self.cal_B = float(cal.get("B", 0.0))
                self.model_version = cal.get("version", self.model_version)
                node_in = self._detect_node_in(self.model_version)
                logger.info("[Inference] Platt: A=%.4f B=%.4f  version=%s detected_node_in=%d",
                            self.cal_A, self.cal_B, self.model_version, node_in)
            except Exception as e:
                logger.warning("[Inference] Could not read calibration: %s", e)

        self.node_in = node_in

        # PyTorch weights — try 3-layer (current) first, fall back to 2-layer (legacy)
        try:
            new_model = DFMGNN(node_in, 64, 1, heads=2, dropout=0.3)
            new_model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            new_model.eval()
            self.model = new_model
            self.available = True
            logger.info("[Inference] Loaded 3-layer DFMGNN (%d-D)", node_in)
        except Exception as e3:
            logger.warning("[Inference] 3-layer load failed: %s -- trying 2-layer legacy", e3)
            try:
                legacy = _Legacy2LayerDFMGNN(node_in, 64, 1, heads=2, dropout=0.3)
                legacy.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                legacy.eval()
                self.model = legacy
                self.available = True
                logger.info("[Inference] Loaded 2-layer legacy DFMGNN (%d-D)", node_in)
            except Exception as e2:
                logger.error("[Inference] Both 3-layer and 2-layer load failed: %s", e2)
                self.available = False
                return

        # ONNX (optional, skip stale 8-D exports)
        self.session = None
        if ONNX_AVAILABLE:
            onnx_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(".onnx") and ".stale" not in f
            ]
            if onnx_files:
                onnx_path = os.path.join(model_dir, onnx_files[0])
                try:
                    sess = ort.InferenceSession(onnx_path)
                    shape = sess.get_inputs()[0].shape
                    feat_dim = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else None
                    if feat_dim is not None and feat_dim != node_in:
                        logger.warning("[Inference] ONNX expects %d-D — disabling", feat_dim)
                    else:
                        self.session = sess
                        logger.info("[Inference] ONNX session loaded")
                except Exception as e:
                    logger.warning("[Inference] ONNX load failed: %s", e)

        # Threshold
        if self._threshold_path and os.path.exists(self._threshold_path):
            try:
                with open(self._threshold_path) as f:
                    self.threshold = float(f.read().strip())
            except Exception:
                pass

        path_label = (
            "ONNX" if self.session else
            "PyTorch" if self.model else "NONE"
        )
        logger.info("[Inference] Active path: %s", path_label)

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, part: PartMetadata) -> dict:
        with self._lock:
            return self._predict_internal(part)

    def _predict_internal(self, part: PartMetadata) -> dict:
        result = self._build_input(part)
        if result is None:
            return {"part_risk_score": 0.0, "face_scores": {}, "engine": "no_faces"}

        x, edge_index, batch = result

        if not self.available:
            return {"part_risk_score": 0.0, "face_scores": {}, "engine": "unavailable"}

        part_score = 0.0
        engine_used = "unavailable"
        raw_logit = 0.0

        # --- ONNX ---
        if self.session is not None:
            try:
                ort_inputs = {
                    "x":          x.cpu().numpy().astype("float32"),
                    "edge_index": edge_index.cpu().numpy(),
                    "batch":      batch.cpu().numpy(),
                }
                ort_outs = self.session.run(None, ort_inputs)
                raw_logit = float(ort_outs[0].item()
                                  if hasattr(ort_outs[0], "item")
                                  else ort_outs[0][0])
                prob = 1.0 / (1.0 + math.exp(-(self.cal_A * raw_logit + self.cal_B)))
                part_score = float(prob)
                engine_used = "onnx"
            except Exception as e:
                logger.error("[Inference] ONNX run failed, falling back: %s", e)

        # --- PyTorch fallback ---
        if engine_used == "unavailable" and self.model is not None:
            try:
                with torch.no_grad():
                    out = self.model(x, edge_index, batch)
                    raw_logit = out.item() if out.dim() == 0 else out[0].item()
                    prob = 1.0 / (1.0 + math.exp(-(self.cal_A * raw_logit + self.cal_B)))
                    part_score = float(prob)
                    engine_used = "pytorch"
            except Exception as e:
                logger.error("[Inference] PyTorch run failed: %s", e)

        # --- Face-level scores ---
        face_scores_dict: dict[str, float] = {}
        if self.model is not None:
            try:
                self.model.eval()
                with torch.no_grad():
                    _, face_scores = self.model.get_face_embeddings_and_scores(
                        x, edge_index
                    )
                    for i, f in enumerate(part.faces):
                        face_scores_dict[f.face_id] = (
                            float(face_scores[i].item())
                            if i < len(face_scores)
                            else 0.0
                        )
            except Exception as e:
                logger.error("[Inference] Face-level scores failed: %s", e)
                for f in part.faces:
                    face_scores_dict[f.face_id] = part_score

        # --- MLflow prediction log (best-effort, never crashes inference) ---
        self._log_prediction_mlflow(
            part_id=part.solidworks_part_number or part.filename,
            n_faces=len(part.faces),
            raw_logit=raw_logit,
            risk_score=part_score,
            engine=engine_used,
        )

        return {
            "part_risk_score": part_score,
            "face_scores": face_scores_dict,
            "engine": engine_used,
            "inference_mode": self.inference_mode,
        }

    # -----------------------------------------------------------------------
    # MLflow
    # -----------------------------------------------------------------------

    @staticmethod
    def _init_mlflow():
        try:
            import mlflow
            _HERE = os.path.dirname(os.path.abspath(__file__))
            _REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
            uri = os.path.join(_REPO_ROOT, "data", "mlflow")
            os.makedirs(uri, exist_ok=True)
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("dfm_gnn_flywheel")
            return mlflow
        except ImportError:
            return None

    def _log_prediction_mlflow(self, part_id: str, n_faces: int,
                               raw_logit: float, risk_score: float,
                               engine: str):
        if self._mlflow is None:
            return
        try:
            with self._mlflow.start_run(
                run_name="inference",
                nested=True,
                tags={"type": "inference"},
            ):
                self._mlflow.log_metrics({
                    "risk_score": risk_score,
                    "raw_logit": raw_logit,
                })
                self._mlflow.log_params({
                    "part_id":       part_id,
                    "model_version": self.model_version,
                    "n_faces":       n_faces,
                    "engine":        engine,
                })
        except Exception:
            pass  # Never let MLflow crash a prediction

    # -----------------------------------------------------------------------
    # Feature builder (unchanged from original)
    # -----------------------------------------------------------------------

    def _build_input(self, part: PartMetadata):
        face_count = len(part.faces)
        if face_count == 0:
            return None

        node_in = getattr(self, "node_in", 12)
        feature_extractor = self.model if self.model is not None else DFMGNN(node_in, 64, 1, heads=2)

        # Graph-level stats are always appended (both 12-D and 14-D models)
        graph_stats = compute_graph_level_stats(part.faces)

        x_list = []
        for f in part.faces:
            feats = feature_extractor.get_node_features(
                f.model_dump() if hasattr(f, "model_dump") else f.__dict__
            )
            # get_node_features returns 10 features for 12-D model, 12 for legacy 14-D
            base_len = len(feats)
            expected_base = node_in - 2  # 10 for 12-D, 12 for 14-D
            if base_len == expected_base:
                feats = feats + graph_stats
            elif base_len + len(graph_stats) == node_in:
                feats = feats + graph_stats
            else:
                # Pad or truncate to match node_in
                feats = (feats + graph_stats)[:node_in]
                while len(feats) < node_in:
                    feats.append(0.0)
            # Assert dynamic dimensions match (e.g. 12-D or 14-D based on model)
            assert len(feats) == node_in, f"Feature dimension mismatch: got {len(feats)}, expected {node_in}"
            x_list.append(feats)

        x = torch.tensor(x_list, dtype=torch.float32)

        edge_list = []
        custom_adj = getattr(part, "adjacency", None)
        if custom_adj is not None:
            for edge in custom_adj:
                if len(edge) == 2:
                    u, v = int(edge[0]), int(edge[1])
                    if 0 <= u < face_count and 0 <= v < face_count:
                        edge_list.append([u, v])
                        edge_list.append([v, u])
        else:
            for i in range(face_count):
                for j in range(i + 1, face_count):
                    edge_list.append([i, j])
                    edge_list.append([j, i])

        edge_index = (
            torch.tensor(edge_list, dtype=torch.long).t()
            if edge_list
            else torch.zeros((2, 0), dtype=torch.long)
        )
        batch = torch.zeros(face_count, dtype=torch.long)
        return x, edge_index, batch

    # -----------------------------------------------------------------------
    # Hybrid GNN+XGBoost
    # -----------------------------------------------------------------------

    def _load_hybrid_model(self):
        """Load XGBoost hybrid model and calibration. Falls back to GNN if missing."""
        if not XGB_AVAILABLE:
            logger.info("[Inference] xgboost not installed -- staying in GNN mode")
            return

        model_dir = os.path.dirname(self._model_path) if self._model_path else ""
        xgb_path = os.path.join(model_dir, "hybrid_xgb.json")
        cal_path = os.path.join(model_dir, "calibration_hybrid.json")

        if not os.path.exists(xgb_path) or not os.path.exists(cal_path):
            logger.info("[Inference] Hybrid model files not found -- staying in GNN mode")
            return

        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)

            with open(cal_path) as f:
                cal = json.load(f)

            self.xgb_model = xgb_model
            self.hybrid_cal_A = float(cal.get("A", 1.0))
            self.hybrid_cal_B = float(cal.get("B", 0.0))
            self.hybrid_embed_dim = int(cal.get("embed_dim", 128))
            self.inference_mode = "hybrid"
            self.model_version = cal.get("version", "hybrid-v1.0")
            logger.info("[Inference] Hybrid XGBoost loaded -- inference_mode='hybrid' "
                        "(embed_dim=%d)", self.hybrid_embed_dim)
        except Exception as e:
            logger.error("[Inference] Failed to load hybrid model: %s -- staying in GNN mode", e)
            self.inference_mode = "gnn"

    def _extract_gnn_embedding(self, x, edge_index, batch):
        """Extract graph-level embedding from GNN (before MLP head)."""
        from torch_geometric.nn import global_mean_pool, global_max_pool
        import torch.nn.functional as F

        model = self.model
        if model is None:
            return None

        with torch.no_grad():
            # Detect architecture by checking for skip connections
            has_skip = hasattr(model, 'skip1')

            if has_skip:
                # 3-layer residual
                h = model.conv1(x, edge_index)
                h = model.bn1(h)
                h = F.leaky_relu(h + model.skip1(x), 0.01)

                h_in = h
                h = model.conv2(h, edge_index)
                h = model.bn2(h)
                h = F.leaky_relu(h + model.skip2(h_in), 0.01)

                h_in = h
                h = model.conv3(h, edge_index)
                h = model.bn3_conv(h)
                h = F.leaky_relu(h + model.skip3(h_in), 0.01)
            else:
                # Legacy 2-layer
                h = model.conv1(x, edge_index)
                h = model.bn1(h)
                h = F.leaky_relu(h, 0.01)

                h = model.conv2(h, edge_index)
                h = model.bn2(h)
                h = F.leaky_relu(h, 0.01)

            mean_pool = global_mean_pool(h, batch)
            max_pool = global_max_pool(h, batch)
            return torch.cat([mean_pool, max_pool], dim=1)

    def _compute_aggregated_features(self, part: PartMetadata):
        """Compute 14-D aggregated part-level features for XGBoost."""
        faces = part.faces
        n = len(faces)
        if n == 0:
            return [0.0] * 14

        thicknesses = [f.thickness_mm for f in faces if f.thickness_mm is not None and f.thickness_mm > 0]
        drafts = [f.draft_angle_deg for f in faces if f.draft_angle_deg is not None]
        radii = [f.radius_mm for f in faces if f.radius_mm is not None and f.radius_mm > 0]
        aspects = []
        for f in faces:
            if f.width_mm and f.depth_mm and f.width_mm > 0:
                aspects.append(min(f.depth_mm / f.width_mm, 1.0) if f.depth_mm else 1.0)
            else:
                aspects.append(1.0)

        face_types = [getattr(f.face_type, "value", str(f.face_type)) for f in faces]
        n_plane = sum(1 for ft in face_types if ft == "Plane")
        n_cyl = sum(1 for ft in face_types if ft == "Cylinder")

        # Normalize to match training scale
        mean_thick = (sum(thicknesses) / len(thicknesses) / 5.0) if thicknesses else 0.0
        min_draft = (min(drafts) / 45.0) if drafts else 0.0
        max_draft = (max(drafts) / 45.0) if drafts else 0.0
        frac_thin = sum(1 for t in thicknesses if t / 5.0 < 0.30) / n if thicknesses else 0.0
        frac_zero = sum(1 for d in drafts if d / 45.0 < 0.012) / n if drafts else 0.0
        min_rad = (min(radii) / 10.0) if radii else 0.0
        mean_asp = sum(aspects) / len(aspects) if aspects else 1.0
        vol_proxy = sum(f.area_mm2 / 10000.0 for f in faces)
        max_thick = (max(thicknesses) / 5.0) if thicknesses else 0.0
        std_thick = 0.0
        if len(thicknesses) > 1:
            m = sum(thicknesses) / len(thicknesses)
            std_thick = (sum((t - m) ** 2 for t in thicknesses) / len(thicknesses)) ** 0.5 / 5.0
        min_thick = (min(thicknesses) / 5.0) if thicknesses else 0.0

        return [
            mean_thick, min_draft, max_draft, frac_thin, frac_zero,
            min_rad, mean_asp, vol_proxy, float(n),
            n_cyl / n, n_plane / n,
            max_thick, std_thick, min_thick,
        ]

    def predict_hybrid(self, part: PartMetadata) -> dict:
        """Hybrid inference: GNN embedding + aggregated features -> XGBoost."""
        with self._lock:
            if self.xgb_model is None or self.model is None:
                return self._predict_internal(part)

            result = self._build_input(part)
            if result is None:
                return {"part_risk_score": 0.0, "face_scores": {}, "engine": "no_faces",
                        "inference_mode": "hybrid"}

            x, edge_index, batch = result

            # 1. GNN embedding
            emb = self._extract_gnn_embedding(x, edge_index, batch)
            if emb is None:
                return self._predict_internal(part)

            emb_np = emb.cpu().numpy().flatten()

            # 2. Aggregated features
            agg = self._compute_aggregated_features(part)

            # 3. Concat and predict
            features = list(emb_np) + agg
            if not XGB_AVAILABLE:
                return self._predict_internal(part)

            X = np.array([features], dtype=np.float32)
            prob = float(self.xgb_model.predict_proba(X)[0, 1])

            # Platt calibration
            logit = math.log(prob / (1.0 - prob + 1e-10) + 1e-10)
            calibrated = 1.0 / (1.0 + math.exp(-(self.hybrid_cal_A * logit + self.hybrid_cal_B)))
            part_score = float(calibrated)

            # Face-level scores still from GNN
            face_scores_dict = {}
            try:
                self.model.eval()
                with torch.no_grad():
                    _, face_scores = self.model.get_face_embeddings_and_scores(x, edge_index)
                    for i, f in enumerate(part.faces):
                        face_scores_dict[f.face_id] = (
                            float(face_scores[i].item()) if i < len(face_scores) else 0.0
                        )
            except Exception:
                for f in part.faces:
                    face_scores_dict[f.face_id] = part_score

            self._log_prediction_mlflow(
                part_id=part.solidworks_part_number or part.filename,
                n_faces=len(part.faces),
                raw_logit=logit,
                risk_score=part_score,
                engine="hybrid_xgb",
            )

            return {
                "part_risk_score": part_score,
                "face_scores": face_scores_dict,
                "engine": "hybrid_xgb",
                "inference_mode": "hybrid",
            }
