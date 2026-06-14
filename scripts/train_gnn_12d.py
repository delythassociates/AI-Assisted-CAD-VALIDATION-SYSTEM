"""Train the 12-D GNN model with:
  - Focal Loss (γ=2, α=0.25)                           [Fix 1]
  - Pairwise AUC ranking loss (weight=0.3)              [Fix 2]
  - Dropped dims 5 (width) and 7 (curv_max) → 12-D     [Fix 3]
  - 3-layer GATv2 with residual skip connections         [Fix 4]
  - Balanced dataset (55/45 split)                       [Fix 5]
  - AUC-gated model saving                               [Fix 6]

Keeps: feature auditing, EMA early stopping, Platt scaling,
       threshold scan, success criteria audit, ONNX export.
"""
import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backend.ml.gnn_model import DFMGNN

MODEL_DIR = "F:/Varroc/data/models"
# [Fix 5] — switched to balanced dataset
DATA_PATH = "F:/Varroc/data/processed/real_cad_training_data_balanced.pt"
OUTPUT_MODEL = os.path.join(MODEL_DIR, "gnn_model_real_cpu.pt")
OUTPUT_THRESHOLD = os.path.join(MODEL_DIR, "threshold_real.txt")
OUTPUT_CALIBRATION = os.path.join(MODEL_DIR, "calibration_real.json")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# [Fix 3] — Columns to drop from 14-D → 12-D
DIMS_TO_DROP = [5, 7]   # width (r=0.91 with depth), curv_max (r=0.97 with curv_min)

# [Fix 1] — Focal Loss (γ=2, α=0.25)
class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced problems.
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    Operates on raw logits (numerically stable via logsigmoid).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# [Fix 2] — Pairwise AUC ranking loss
def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor,
                          margin: float = 1.0) -> torch.Tensor:
    """Sample (defective, clean) pairs and penalize when defective scores below clean.
    Uses a differentiable hinge-like formulation:
       loss = max(0, margin - (logit_pos - logit_neg))
    averaged over all valid pairs in the batch.
    """
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]

    if len(pos_logits) == 0 or len(neg_logits) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Cap sampled pairs to avoid O(n²) explosion on large batches
    max_pairs = 256
    n_pos = len(pos_logits)
    n_neg = len(neg_logits)

    if n_pos * n_neg > max_pairs:
        idx_pos = torch.randint(n_pos, (max_pairs,), device=logits.device)
        idx_neg = torch.randint(n_neg, (max_pairs,), device=logits.device)
    else:
        # All pairs (grid)
        idx_pos = torch.arange(n_pos, device=logits.device).repeat_interleave(n_neg)
        idx_neg = torch.arange(n_neg, device=logits.device).repeat(n_pos)

    diff = pos_logits[idx_pos] - neg_logits[idx_neg]
    loss = F.relu(margin - diff)
    return loss.mean()


def strip_dims(x: torch.Tensor, dims_to_drop: list[int]) -> torch.Tensor:
    """Drop specified column indices from a 2-D tensor."""
    keep = [i for i in range(x.shape[1]) if i not in dims_to_drop]
    return x[:, keep]


def compute_batch_auc(logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Safe AUC-ROC, returns 0.5 if only one class present."""
    if len(set(labels_np)) < 2:
        return 0.5
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    return roc_auc_score(labels_np, probs)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("EUREKA 3.0 — 12-D GNN (Focal + Ranking + 3-Layer Residual)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}.")
        print("Run balance_dataset.py first.")
        sys.exit(1)

    graphs = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    print(f"Loaded {len(graphs)} graphs from {DATA_PATH}")

    # Filter invalid graphs (must have ≥3 nodes)
    valid_graphs = [g for g in graphs if g.x.shape[0] >= 3]
    print(f"Valid graphs (>=3 faces): {len(valid_graphs)}/{len(graphs)}")

    # [Fix 3] — Drop redundant features in-place
    original_dim = valid_graphs[0].x.shape[1]
    for g in valid_graphs:
        g.x = strip_dims(g.x, DIMS_TO_DROP)
    NODE_IN = valid_graphs[0].x.shape[1]
    print(f"Feature dims: {original_dim}-D -> {NODE_IN}-D (dropped dims {DIMS_TO_DROP})")

    # 1. Feature Audit
    print(f"\n--- Auditing {NODE_IN}-D Features ---")
    all_x = torch.cat([g.x for g in valid_graphs], dim=0).numpy()

    feature_names = [
        "face_type", "thickness", "radius", "area", "depth",
        # (width dropped)
        "curv_min",
        # (curv_max dropped)
        "draft_angle", "parent_wall",
        "centroid_dist", "aspect_ratio", "min_plane_draft", "frac_plane"
    ]
    # Adjust if dataset had fewer dims (e.g. 12-D base without graph stats)
    for i in range(min(NODE_IN, len(feature_names))):
        col = all_x[:, i]
        name = feature_names[i] if i < len(feature_names) else f"dim_{i}"
        print(f"{name}: mean={col.mean():.3f}, std={col.std():.3f}, "
              f"% saturated at 1.0={(col >= 1.0).mean()*100:.1f}%")

    corr_matrix = np.corrcoef(all_x, rowvar=False)
    print("\nHighly Correlated Feature Pairs (> 0.85):")
    found_high_corr = False
    for i in range(NODE_IN):
        for j in range(i + 1, NODE_IN):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.85:
                ni = feature_names[i] if i < len(feature_names) else f"dim_{i}"
                nj = feature_names[j] if j < len(feature_names) else f"dim_{j}"
                print(f"  Feature {i} ({ni}) <-> Feature {j} ({nj}): {corr:.4f}")
                found_high_corr = True
    if not found_high_corr:
        print("  No highly correlated feature pairs found.")

    # 2. Class Imbalance Diagnostic
    print("\n--- Class Distribution ---")
    y_all = np.array([g.y.item() for g in valid_graphs])
    pos_count = int(np.sum(y_all == 1))
    neg_count = int(np.sum(y_all == 0))
    print(f"Defective (Positive) = {pos_count} | Clean (Negative) = {neg_count}")
    print(f"Ratio: {pos_count/max(neg_count,1):.2f}:1")

    # Split dataset  (70 / 15 / 15)
    dataset = valid_graphs
    random.shuffle(dataset)
    n = len(dataset)
    train_dataset = dataset[:int(0.7*n)]
    val_dataset   = dataset[int(0.7*n):int(0.85*n)]
    test_dataset  = dataset[int(0.85*n):]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # [Fix 4] — 3-layer GATv2 with residual skips (defined in gnn_model.py)
    model = DFMGNN(node_in=NODE_IN, hidden=64, out_dim=1, heads=2, dropout=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)

    # [Fix 1] — Focal Loss
    focal_criterion = FocalLoss(gamma=2.0, alpha=0.25)
    RANKING_WEIGHT = 0.3  # [Fix 2] weight for ranking loss

    # CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)

    # [Fix 6] — AUC-gated early stopping (track val AUC, not val loss)
    best_val_auc = 0.0
    best_state = None
    best_epoch = 1
    patience = 40
    patience_counter = 0
    min_epochs = 35

    print(f"\nStarting training (Focal gamma=2 alpha=0.25, ranking_w={RANKING_WEIGHT})...")
    for epoch in range(150):
        model.train()
        total_focal = 0.0
        total_rank = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y

            # [Fix 1] Focal loss
            fl = focal_criterion(logits, labels)

            # [Fix 2] Pairwise ranking loss
            rl = pairwise_ranking_loss(logits, labels)

            loss = fl + RANKING_WEIGHT * rl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_focal += fl.item()
            total_rank += rl.item()

        n_batches = max(len(train_loader), 1)

        # --- Validation: compute AUC ---
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                val_logits_list.extend(logits.cpu().numpy())
                val_labels_list.extend(batch.y.cpu().numpy())

        val_logits_arr = np.array(val_logits_list)
        val_labels_arr = np.array(val_labels_list)
        val_auc = compute_batch_auc(val_logits_arr, val_labels_arr)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits_arr))
        val_preds = (val_probs > 0.5).astype(int)
        val_acc = accuracy_score(val_labels_arr, val_preds)

        scheduler.step()

        # [Fix 6] — checkpoint on AUC improvement, not loss
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            if epoch >= min_epochs:
                patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Focal: {total_focal/n_batches:.4f} "
                  f"| Rank: {total_rank/n_batches:.4f} "
                  f"| Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f}) "
                  f"| Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch}, best AUC: {best_val_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)

    # --- Platt Scaling Calibration Fit ---
    print("\n--- Fitting Platt Scaling Calibration ---")
    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            val_logits_list.extend(logits.cpu().numpy())
            val_labels_list.extend(batch.y.cpu().numpy())

    val_logits_arr = np.array(val_logits_list).reshape(-1, 1)
    val_labels_arr = np.array(val_labels_list)

    calibrator = LogisticRegression(C=999999.0)
    calibrator.fit(val_logits_arr, val_labels_arr)

    A = calibrator.coef_[0][0]
    B = calibrator.intercept_[0]
    print(f"Platt Scaling parameters fitted: A={A:.4f}, B={B:.4f}")

    # --- Test Evaluation ---
    model.eval()
    test_logits, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            test_logits.extend(logits.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())

    test_logits_arr = np.array(test_logits)
    test_labels_arr = np.array(test_labels)

    # Calibrate test logits using Platt scaling
    test_probs_calibrated = 1.0 / (1.0 + np.exp(-(A * test_logits_arr + B)))
    test_preds_calibrated = (test_probs_calibrated >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(test_labels_arr, test_probs_calibrated) if len(set(test_labels_arr)) > 1 else 0.5
    f1_05 = f1_score(test_labels_arr, test_preds_calibrated) if len(set(test_labels_arr)) > 1 else 0.0

    print(f"\n=== Test Results (calibrated, threshold=0.5) ===")
    print(f"  Accuracy:            {accuracy_score(test_labels_arr, test_preds_calibrated):.4f}")
    if len(set(test_labels_arr)) > 1:
        print(f"  Precision:           {precision_score(test_labels_arr, test_preds_calibrated):.4f}")
        print(f"  Recall:              {recall_score(test_labels_arr, test_preds_calibrated):.4f}")
        print(f"  F1 Score @ 0.5:      {f1_05:.4f}")
        print(f"  AUC-ROC:             {auc:.4f}")

        # Optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(test_labels_arr, test_probs_calibrated)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        f1_opt = f1_scores[best_idx]
        print(f"  F1 Score @ optimal:  {f1_opt:.4f} (optimal threshold: {best_threshold:.4f})")
    else:
        f1_opt = 0.0
        best_threshold = 0.5
        print("  Skipped: only one class in test split.")

    opt_thresh = best_threshold
    cal_A = A
    cal_B = B
    test_labels = test_labels_arr
    test_preds_05 = test_preds_calibrated

    # Retrain Report
    print("\n=== Retrain Report ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val AUC: {best_val_auc:.4f}")
    print(f"Test AUC-ROC: {auc:.4f}")
    print(f"F1 @ 0.5: {f1_05:.4f}")
    print(f"F1 @ optimal threshold: {f1_opt:.4f}")
    print(f"Optimal threshold: {opt_thresh:.4f}")
    print(f"Platt A: {cal_A:.4f}, B: {cal_B:.4f}")

    # Confusion matrix
    print("\n=== Confusion Matrix (Threshold 0.5) ===")
    print(confusion_matrix(test_labels, test_preds_05))
    print("\n=== Classification Report (Threshold 0.5) ===")
    print(classification_report(test_labels, test_preds_05,
          target_names=["Clean", "Defective"]))

    # Threshold awareness scan
    print("\n--- Threshold Awareness Scan ---")
    probs = test_probs_calibrated
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        preds = (probs >= thresh).astype(float)
        cm_t = confusion_matrix(test_labels, preds)
        clean_recall_t = cm_t[0,0] / cm_t[0].sum() if cm_t[0].sum() > 0 else 0.0
        def_recall_t   = cm_t[1,1] / cm_t[1].sum() if cm_t[1].sum() > 0 else 0.0
        print(f"Thresh {thresh:.2f} | Clean recall: {clean_recall_t:.3f} "
              f"| Defective recall: {def_recall_t:.3f} "
              f"| F1: {f1_score(test_labels, preds):.4f}")

    # Success Criteria Audit
    cm_05 = confusion_matrix(test_labels, test_preds_05)
    clean_recall_05 = cm_05[0, 0] / cm_05[0].sum() if cm_05[0].sum() > 0 else 0.0
    def_recall_05 = cm_05[1, 1] / cm_05[1].sum() if cm_05[1].sum() > 0 else 0.0

    met_auc = auc > 0.64
    met_f1 = f1_05 > 0.80
    met_clean_rec = clean_recall_05 > 0.30
    met_def_rec = def_recall_05 > 0.85
    met_opt_thresh = 0.30 <= opt_thresh <= 0.55

    print("\n=== Success Criteria Audit ===")
    print(f"  AUC-ROC > 0.64:               {auc:.4f} ( {'Met' if met_auc else 'FAILED'} )")
    print(f"  F1 @ 0.5 > 0.80:              {f1_05:.4f} ( {'Met' if met_f1 else 'FAILED'} )")
    print(f"  Clean Recall @ 0.5 > 0.30:    {clean_recall_05:.4f} ( {'Met' if met_clean_rec else 'FAILED'} )")
    print(f"  Defective Recall @ 0.5 > 0.85: {def_recall_05:.4f} ( {'Met' if met_def_rec else 'FAILED'} )")
    print(f"  Optimal Threshold 0.30-0.55:  {opt_thresh:.4f} ( {'Met' if met_opt_thresh else 'FAILED'} )")

    # Calibration curve plot
    try:
        prob_true, prob_pred = calibration_curve(test_labels, test_probs_calibrated, n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model (Calibrated)')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve (Platt Scaled)')
        plt.legend()
        curve_path = os.path.join(MODEL_DIR, "calibration_curve.png")
        plt.savefig(curve_path)
        plt.close()
        print(f"Calibration curve saved to {curve_path}")
    except Exception as plt_err:
        print(f"Failed to plot calibration curve: {plt_err}")

    # [Fix 6] — Export gated on AUC > 0.83
    AUC_EXPORT_GATE = 0.83
    export_criteria_met = auc > AUC_EXPORT_GATE
    if not export_criteria_met:
        print(f"\n[!] AUC {auc:.4f} < {AUC_EXPORT_GATE} gate -- model NOT exported.")
        print("   Existing weights were NOT overwritten.")
    else:
        print(f"\n[OK] AUC {auc:.4f} > {AUC_EXPORT_GATE} gate -- exporting model.")
        torch.save(model.state_dict(), OUTPUT_MODEL)
        with open(OUTPUT_THRESHOLD, "w") as f:
            f.write("0.5")
        # [Fix 5] — update calibration version to reflect 12-D
        with open(OUTPUT_CALIBRATION, "w") as f:
            json.dump({"A": float(cal_A), "B": float(cal_B),
                       "version": "v2.0-12d-focal-rank"}, f, indent=2)
        print("Model + calibration exported.")

        # ONNX export
        try:
            model_cpu = model.cpu()
            model_cpu.eval()
            dummy_x = torch.randn(1, NODE_IN)
            dummy_edge = torch.zeros(2, 0, dtype=torch.long)
            dummy_batch = torch.zeros(1, dtype=torch.long)
            onnx_path = os.path.join(MODEL_DIR, "gnn_model_real.onnx")

            torch.onnx.export(
                model_cpu,
                (dummy_x, dummy_edge, dummy_batch),
                onnx_path,
                input_names=["x", "edge_index", "batch"],
                output_names=["risk_score_logits"],
                dynamic_axes={
                    "x": {0: "num_nodes"},
                    "edge_index": {1: "num_edges"},
                    "batch": {0: "num_nodes"},
                },
                opset_version=17
            )
            print(f"ONNX model exported successfully: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
