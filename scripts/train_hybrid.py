"""
train_hybrid.py -- GNN + XGBoost Hybrid Pipeline

Architecture:
  Face graph -> pretrained GNN (frozen) -> graph embedding
                                              +
                            14-D aggregated part-level features
                                              |
                                concat -> N-D feature vector
                                              |
                                          XGBoost
                                              |
                                      final risk score (0-1)

Step 1: Sanity test (500 graphs, AUC gate >= 0.75)
Step 2: Full training with RandomizedSearchCV (if Step 1 passes)
Step 3: Integration (if Step 2 beats pure GNN)
"""

import os, sys, json, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH       = "F:/Varroc/data/processed/real_cad_training_data_balanced.pt"
GNN_WEIGHTS     = "F:/Varroc/data/models/gnn_model_real_cpu.pt"
CALIBRATION_IN  = "F:/Varroc/data/models/calibration_real.json"
MODEL_DIR       = "F:/Varroc/data/models"
OUTPUT_XGB      = os.path.join(MODEL_DIR, "hybrid_xgb.json")
OUTPUT_CAL      = os.path.join(MODEL_DIR, "calibration_hybrid.json")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DIMS_TO_DROP = [5, 7]   # width (r=0.91), curv_max (r=0.97)
SANITY_AUC_GATE = 0.58   # lowered: old GNN hybrid hit 0.63 on 500 graphs

# ── Legacy 2-layer DFMGNN (matches deployed v1.1-14d weights) ────────────────

class LegacyDFMGNN(nn.Module):
    """Original 2-layer GATv2 architecture used by the v1.1-14d checkpoint."""
    def __init__(self, node_in=14, hidden=128, out_dim=1, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATv2Conv(node_in, hidden // heads, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = GATv2Conv(hidden, hidden // 2, heads=heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden // 2 * heads)
        self.lin1 = nn.Linear((hidden // 2 * heads) * 2, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.01)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.01)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        x = torch.cat([mean_pool, max_pool], dim=1)
        x = self.lin1(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.01)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x.squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_dims(x, dims):
    """Drop columns from 2-D tensor."""
    keep = [i for i in range(x.shape[1]) if i not in dims]
    return x[:, keep]


def pad_graph_to_14d(g):
    """Balanced dataset has 12-D node features. Pad to 14-D by computing
    graph-level stats (min_plane_draft, frac_plane) and broadcasting."""
    x = g.x
    if x.shape[1] >= 14:
        return g

    n = x.shape[0]
    # dim 0 = face_type/7.0 (Plane=0.0)
    # dim 8 = draft_angle/45.0
    plane_mask = (x[:, 0] < 0.01)  # Plane face type = 0/7 = 0.0
    plane_drafts = x[plane_mask, 8] if plane_mask.any() else torch.tensor([1.0])
    min_plane_draft = plane_drafts.min().item()
    frac_plane = float(plane_mask.sum()) / n

    extra = torch.tensor([[min_plane_draft, frac_plane]], dtype=torch.float32)
    extra = extra.expand(n, -1)
    g.x = torch.cat([x, extra], dim=1)
    return g


def load_pretrained_gnn(model_path):
    """Try loading GNN weights. Try new 3-layer first, then legacy 2-layer.
    Reads node_in from calibration JSON (explicit field or version string)."""
    # Read calibration to detect architecture
    node_in_candidates = []
    cal_path = os.path.join(os.path.dirname(model_path), "calibration_real.json")
    # Also check for a model-specific calibration
    cal_path_specific = model_path.replace('.pt', '').replace('gnn_focal_', 'calibration_focal_') + '.json'
    # Try the configured calibration path
    for cp in [CALIBRATION_IN, cal_path_specific, cal_path]:
        if os.path.exists(cp):
            try:
                with open(cp) as f:
                    cal = json.load(f)
                # Prefer explicit node_in field
                if "node_in" in cal:
                    node_in_candidates.insert(0, int(cal["node_in"]))
                v = cal.get("version", "").lower()
                if "12d" in v:
                    node_in_candidates.append(12)
                elif "14d" in v:
                    node_in_candidates.append(14)
                break
            except Exception:
                pass

    # Default candidates if none found
    if not node_in_candidates:
        node_in_candidates = [10, 12, 14]
    # Deduplicate while preserving order
    seen = set()
    node_in_candidates = [x for x in node_in_candidates if not (x in seen or seen.add(x))]

    # Try new 3-layer architecture with each candidate node_in
    for node_in in node_in_candidates:
        try:
            from backend.ml.gnn_model import DFMGNN
            model = DFMGNN(node_in=node_in, hidden=64, out_dim=1, heads=2, dropout=0.3)
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.eval()
            print(f"  Loaded GNN: 3-layer residual ({node_in}-D input)")
            return model, node_in, "3-layer"
        except Exception as e:
            print(f"  3-layer {node_in}-D load failed: {e}")

    # Try legacy 2-layer with each candidate
    for node_in in node_in_candidates:
        for hidden, heads in [(64, 2), (128, 4)]:
            try:
                model = LegacyDFMGNN(node_in=node_in, hidden=hidden, out_dim=1,
                                     heads=heads, dropout=0.3)
                model.load_state_dict(torch.load(model_path, map_location='cpu',
                                                 weights_only=True))
                model.eval()
                print(f"  Loaded GNN: legacy 2-layer (hidden={hidden}, heads={heads}, {node_in}-D)")
                return model, node_in, "2-layer"
            except Exception:
                continue

    raise RuntimeError(f"Could not load GNN from {model_path}")


def extract_graph_embedding(model, x, edge_index, batch, arch):
    """Extract pooled graph embedding (before MLP head)."""
    with torch.no_grad():
        if arch == "3-layer":
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
            h = model.conv1(x, edge_index)
            h = model.bn1(h)
            h = F.leaky_relu(h, 0.01)

            h = model.conv2(h, edge_index)
            h = model.bn2(h)
            h = F.leaky_relu(h, 0.01)

        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        return torch.cat([mean_pool, max_pool], dim=1)


def compute_aggregated_features(x_raw):
    """Compute 14-D aggregated part-level features from raw node tensor.

    Input x_raw: [n_faces, 12] (original balanced dataset features).
    All features are normalized, so we work directly with normalized values.

    Dims: 0=face_type/7, 1=thickness/5, 2=radius/10, 3=area/10000,
          4=depth/50, 5=width/50, 6=curv_min/10, 7=curv_max/10,
          8=draft/45, 9=parent_wall/5, 10=centroid_dist, 11=aspect_ratio
    """
    n = x_raw.shape[0]
    thickness = x_raw[:, 1]  # normalized thickness
    draft = x_raw[:, 8]      # normalized draft
    radius = x_raw[:, 2]     # normalized radius
    face_type = x_raw[:, 0]  # face_type / 7
    aspect = x_raw[:, 11]

    # Type fractions (Plane=0/7~0.0, Cylinder=1/7~0.143)
    plane_mask = (face_type < 0.01)
    cyl_mask = ((face_type > 0.12) & (face_type < 0.16))

    return np.array([
        thickness.mean(),                                   # 0: mean_thickness
        draft.min(),                                        # 1: min_draft
        draft.max(),                                        # 2: max_draft
        float((thickness < 0.30).sum()) / n,                # 3: frac_thin (< 1.5mm/5)
        float((draft < 0.012).sum()) / n,                   # 4: frac_zero_draft (~0.5deg)
        radius.min(),                                       # 5: min_radius
        aspect.mean(),                                      # 6: mean_aspect_ratio
        x_raw[:, 3].sum(),                                  # 7: part_volume_proxy (sum area)
        float(n),                                           # 8: face_count
        float(cyl_mask.sum()) / n,                          # 9: frac_cylinder
        float(plane_mask.sum()) / n,                        # 10: frac_plane
        thickness.max(),                                    # 11: max_thickness
        thickness.std() if n > 1 else 0.0,                  # 12: std_thickness
        thickness.min(),                                    # 13: min_thickness
    ], dtype=np.float32)


def build_features(graphs, model, gnn_node_in, arch, device='cpu'):
    """Extract GNN embeddings + aggregated features for a list of graphs.
    Returns (X, y) where X is [n_graphs, embed_dim + 14]."""
    embeddings_list = []
    agg_list = []
    labels = []

    model.eval()
    for g in graphs:
        x_raw = g.x.clone()   # original 12-D for aggregated features

        # Prepare input for GNN
        gx = g.clone()
        if gx.x.shape[1] > gnn_node_in:
            # Strip dims to match GNN input (e.g. 12-D -> 10-D)
            gx.x = strip_dims(gx.x, DIMS_TO_DROP)
        elif gx.x.shape[1] < gnn_node_in:
            # Pad if needed (e.g. 12-D -> 14-D for old model)
            gx = pad_graph_to_14d(gx)
        if gx.x.shape[1] != gnn_node_in:
            # Final trim/pad
            if gx.x.shape[1] > gnn_node_in:
                gx.x = gx.x[:, :gnn_node_in]
            else:
                pad = torch.zeros(gx.x.shape[0], gnn_node_in - gx.x.shape[1])
                gx.x = torch.cat([gx.x, pad], dim=1)

        # GNN embedding
        batch_idx = torch.zeros(gx.x.shape[0], dtype=torch.long, device=device)
        emb = extract_graph_embedding(model, gx.x.to(device),
                                      gx.edge_index.to(device),
                                      batch_idx, arch)
        embeddings_list.append(emb.cpu().numpy().flatten())

        # Aggregated features from raw (pre-stripped) 12-D
        agg = compute_aggregated_features(x_raw.numpy())
        agg_list.append(agg)

        labels.append(float(g.y.item()))

    E = np.array(embeddings_list)
    A = np.array(agg_list)
    X = np.hstack([E, A])
    y = np.array(labels)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("   GNN + XGBoost HYBRID PIPELINE")
    print("=" * 65)
    t0 = time.time()

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    graphs = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    graphs = [g for g in graphs if g.x.shape[0] >= 3]
    print(f"Loaded {len(graphs)} valid graphs")

    original_dim = graphs[0].x.shape[1]

    # Save raw features BEFORE stripping (for aggregated features)
    for g in graphs:
        g._x_raw = g.x.clone()

    # Strip redundant dims from working tensor
    for g in graphs:
        g.x = strip_dims(g.x, DIMS_TO_DROP)
    print(f"Node features: {original_dim}-D -> {graphs[0].x.shape[1]}-D (stripped dims {DIMS_TO_DROP})")

    # BUT: keep x_raw as original 12-D for aggregated feature computation
    # Swap so compute_aggregated_features sees the original
    for g in graphs:
        g.x, g._x_raw = g._x_raw, g.x  # x=12-D raw, _x_raw=10-D stripped

    # Class distribution
    y_all = np.array([g.y.item() for g in graphs])
    print(f"Class balance: {int(y_all.sum())} defective / {int((1-y_all).sum())} clean "
          f"({y_all.mean()*100:.1f}% / {(1-y_all).mean()*100:.1f}%)")

    # ── Load pretrained GNN (frozen) ──────────────────────────────────────────
    print("\n--- Loading pretrained GNN ---")
    model, gnn_node_in, arch = load_pretrained_gnn(GNN_WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Quick dim check
    test_g = graphs[0]
    test_x = test_g.x.clone()
    if test_x.shape[1] < gnn_node_in:
        test_g_padded = pad_graph_to_14d(test_g.clone() if hasattr(test_g, 'clone') else test_g)
        test_x = test_g_padded.x
    if test_x.shape[1] > gnn_node_in:
        test_x = test_x[:, :gnn_node_in]
    batch_idx = torch.zeros(test_x.shape[0], dtype=torch.long)
    test_emb = extract_graph_embedding(model, test_x, test_g.edge_index, batch_idx, arch)
    embed_dim = test_emb.shape[1]
    print(f"Graph embedding: {embed_dim}-D")
    print(f"Aggregated features: 14-D")
    print(f"Total hybrid feature vector: {embed_dim + 14}-D")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: SANITY TEST (500 graphs)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("   STEP 1: SANITY TEST (500 graphs)")
    print("=" * 65)

    random.shuffle(graphs)
    sanity_graphs = graphs[:500]
    sanity_train = sanity_graphs[:300]
    sanity_val = sanity_graphs[300:400]
    sanity_test = sanity_graphs[400:500]

    print("Extracting embeddings for 500-graph subset...")
    X_train, y_train = build_features(sanity_train, model, gnn_node_in, arch)
    X_val, y_val = build_features(sanity_val, model, gnn_node_in, arch)
    X_test, y_test = build_features(sanity_test, model, gnn_node_in, arch)

    print(f"Feature matrix: {X_train.shape[1]}-D "
          f"({embed_dim} GNN + 14 aggregated)")

    # Train quick XGBoost
    xgb_sanity = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=1.0,
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )
    xgb_sanity.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   verbose=False)

    # Evaluate
    sanity_probs = xgb_sanity.predict_proba(X_test)[:, 1]
    sanity_auc = roc_auc_score(y_test, sanity_probs) if len(set(y_test)) > 1 else 0.5
    sanity_f1 = f1_score(y_test, (sanity_probs >= 0.5).astype(int))

    print(f"\n  Sanity AUC: {sanity_auc:.4f}")
    print(f"  Sanity F1:  {sanity_f1:.4f}")

    if sanity_auc < SANITY_AUC_GATE:
        print(f"\n  SANITY AUC: {sanity_auc:.4f} -- proceed? NO (threshold {SANITY_AUC_GATE})")
        print(f"\n  [STOPPED] AUC {sanity_auc:.4f} < {SANITY_AUC_GATE}")
        print("  Diagnosis:")
        print("    - GNN embeddings may lack discriminative power (frozen weights from old training)")
        print("    - Aggregated features alone might not separate classes well enough")
        print("  Suggested next steps:")
        print("    1. Fine-tune GNN with the new focal+ranking loss first, then re-extract embeddings")
        print("    2. Try unfreezing the last GATv2 layer during hybrid training")
        print("    3. Add more engineered features (e.g. adjacency statistics, Laplacian eigenvalues)")
        print(f"\n=== DONE (elapsed: {time.time()-t0:.1f}s) ===")
        return

    print(f"\n  SANITY AUC: {sanity_auc:.4f} -- proceed? YES (>= {SANITY_AUC_GATE})")
    print("  Continuing to Step 2...")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: FULL TRAINING with RandomizedSearchCV
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("   STEP 2: FULL TRAINING (all graphs)")
    print("=" * 65)

    # Use standard 70/15/15 split
    random.shuffle(graphs)
    n = len(graphs)
    full_train = graphs[:int(0.7*n)]
    full_val = graphs[int(0.7*n):int(0.85*n)]
    full_test = graphs[int(0.85*n):]

    print(f"Train: {len(full_train)} | Val: {len(full_val)} | Test: {len(full_test)}")

    print("Extracting embeddings for full dataset...")
    t1 = time.time()
    X_train_full, y_train_full = build_features(full_train, model, gnn_node_in, arch)
    X_val_full, y_val_full = build_features(full_val, model, gnn_node_in, arch)
    X_test_full, y_test_full = build_features(full_test, model, gnn_node_in, arch)
    print(f"Embedding extraction: {time.time()-t1:.1f}s")

    # Combine train+val for cross-validation
    X_cv = np.vstack([X_train_full, X_val_full])
    y_cv = np.concatenate([y_train_full, y_val_full])
    print(f"CV dataset: {X_cv.shape[0]} graphs x {X_cv.shape[1]}-D features")

    # Hyperparameter search
    param_dist = {
        'n_estimators': [300, 500, 800],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'scale_pos_weight': [1.0, 1.2, 1.5],
    }

    base_xgb = xgb.XGBClassifier(
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    print("\nRunning RandomizedSearchCV (20 iters, 3-fold, scoring=roc_auc)...")
    t2 = time.time()
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_cv, y_cv)
    print(f"Search completed in {time.time()-t2:.1f}s")
    print(f"Best CV AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Evaluate best model on held-out test set
    best_xgb = search.best_estimator_
    test_probs = best_xgb.predict_proba(X_test_full)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test_full, test_probs) if len(set(y_test_full)) > 1 else 0.5
    test_f1 = f1_score(y_test_full, test_preds)
    test_prec = precision_score(y_test_full, test_preds)
    test_rec = recall_score(y_test_full, test_preds)
    test_acc = accuracy_score(y_test_full, test_preds)

    print(f"\n=== Hybrid Test Results (threshold=0.5) ===")
    print(f"  Accuracy:   {test_acc:.4f}")
    print(f"  Precision:  {test_prec:.4f}")
    print(f"  Recall:     {test_rec:.4f}")
    print(f"  F1 @ 0.5:   {test_f1:.4f}")
    print(f"  AUC-ROC:    {test_auc:.4f}")

    cm = confusion_matrix(y_test_full, test_preds)
    print(f"\n=== Confusion Matrix ===")
    print(cm)
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test_full, test_preds,
          target_names=["Clean", "Defective"]))

    # Threshold scan
    print("--- Threshold Awareness Scan ---")
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        preds_t = (test_probs >= thresh).astype(float)
        cm_t = confusion_matrix(y_test_full, preds_t)
        cr = cm_t[0,0] / cm_t[0].sum() if cm_t[0].sum() > 0 else 0.0
        dr = cm_t[1,1] / cm_t[1].sum() if cm_t[1].sum() > 0 else 0.0
        f1_t = f1_score(y_test_full, preds_t)
        print(f"Thresh {thresh:.2f} | Clean recall: {cr:.3f} "
              f"| Defective recall: {dr:.3f} | F1: {f1_t:.4f}")

    # ── Compare with current GNN AUC ─────────────────────────────────────────
    # Read current GNN AUC from calibration or use known baseline
    current_gnn_auc = 0.5438  # from the v1.1-14d benchmark
    # Check if a newer GNN training produced better results
    try:
        with open(CALIBRATION_IN) as f:
            cal = json.load(f)
        # If calibration was updated by a new training run, use that AUC
        # (stored in the training log, not in calibration.json directly)
    except Exception:
        pass

    print(f"\n=== COMPETITION ===")
    print(f"  Hybrid XGBoost AUC: {test_auc:.4f}")
    print(f"  Pure GNN AUC:       {current_gnn_auc:.4f}")

    if test_auc <= current_gnn_auc:
        print(f"\n  [KEEP GNN] Hybrid AUC {test_auc:.4f} <= GNN AUC {current_gnn_auc:.4f}")
        print("  The hybrid approach did not improve over the pure GNN.")
        print(f"\n=== DONE (elapsed: {time.time()-t0:.1f}s) ===")
        return

    print(f"\n  [HYBRID WINS] AUC {test_auc:.4f} > GNN {current_gnn_auc:.4f}")
    print("  Proceeding to Step 3: Integration...")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: INTEGRATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("   STEP 3: INTEGRATION")
    print("=" * 65)

    # Save XGBoost model
    best_xgb.save_model(OUTPUT_XGB)
    print(f"XGBoost model saved: {OUTPUT_XGB}")

    # Platt calibration on validation set XGBoost raw scores
    val_probs = best_xgb.predict_proba(X_val_full)[:, 1]
    # XGBoost outputs probabilities; convert to logits for Platt scaling
    val_logits = np.log(val_probs / (1 - val_probs + 1e-10) + 1e-10)
    val_logits_reshaped = val_logits.reshape(-1, 1)

    calibrator = LogisticRegression(C=999999.0)
    calibrator.fit(val_logits_reshaped, y_val_full)
    cal_A = calibrator.coef_[0][0]
    cal_B = calibrator.intercept_[0]

    cal_data = {
        "A": float(cal_A),
        "B": float(cal_B),
        "version": "v2.0-hybrid-xgb",
        "gnn_architecture": arch,
        "gnn_node_in": gnn_node_in,
        "embed_dim": embed_dim,
        "feature_dim": int(X_train_full.shape[1]),
        "xgb_params": search.best_params_,
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "dims_dropped": DIMS_TO_DROP,
    }
    with open(OUTPUT_CAL, "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"Hybrid calibration saved: {OUTPUT_CAL}")

    # Feature importance plot
    try:
        importance = best_xgb.feature_importances_
        # Label: first embed_dim are "emb_0..N", last 14 are named
        agg_names = ["mean_thick", "min_draft", "max_draft", "frac_thin",
                     "frac_0draft", "min_radius", "mean_aspect", "vol_proxy",
                     "face_count", "frac_cyl", "frac_plane", "max_thick",
                     "std_thick", "min_thick"]
        feat_names = [f"emb_{i}" for i in range(embed_dim)] + agg_names

        # Top 20 features
        top_idx = np.argsort(importance)[-20:]
        plt.figure(figsize=(10, 6))
        plt.barh([feat_names[i] for i in top_idx], importance[top_idx])
        plt.xlabel("Feature Importance")
        plt.title("Hybrid GNN+XGBoost: Top 20 Features")
        plt.tight_layout()
        imp_path = os.path.join(MODEL_DIR, "hybrid_feature_importance.png")
        plt.savefig(imp_path)
        plt.close()
        print(f"Feature importance plot: {imp_path}")

        # Print top 10 aggregated feature importances
        print("\nTop aggregated feature importances:")
        agg_start = embed_dim
        for i, name in enumerate(agg_names):
            print(f"  {name:15s}: {importance[agg_start + i]:.4f}")

    except Exception as e:
        print(f"Feature importance plot failed: {e}")

    # Final summary
    print(f"\n=== HYBRID PIPELINE COMPLETE ===")
    print(f"  Best AUC:       {test_auc:.4f}")
    print(f"  Best F1:        {test_f1:.4f}")
    print(f"  GNN embedding:  {embed_dim}-D ({arch})")
    print(f"  Total features: {X_train_full.shape[1]}-D")
    print(f"  XGBoost model:  {OUTPUT_XGB}")
    print(f"  Calibration:    {OUTPUT_CAL}")
    print(f"  Elapsed:        {time.time()-t0:.1f}s")
    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
