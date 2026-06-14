"""Retrain 12-D GAT GNN on existing 8-D training data (feature padding)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from backend.ml.gnn_model import DFMGNN

MODEL_DIR = "F:/Varroc/data/models"
DATA_PATHS = [
    "F:/Varroc/data/processed/real_training_data.pt",
    "F:/Varroc/data/processed/training_data.pt",
]
OUTPUT_MODEL = os.path.join(MODEL_DIR, "gnn_model_real_cpu.pt")
OUTPUT_THRESHOLD = os.path.join(MODEL_DIR, "threshold_real.txt")

torch.manual_seed(42)
np.random.seed(42)


def pad_features_8_to_12(x_8d):
    """Pad 8-D features to 12-D with sensible defaults for missing fields."""
    batch_dim = x_8d.dim() == 3
    if batch_dim:
        B, N, F = x_8d.shape
        x_12d = x_8d.new_zeros(B, N, 12)
        x_12d[:, :, :8] = x_8d
        x_12d[:, :, 8] = 0.0   # draft angle (default: 0 for synthetic)
        x_12d[:, :, 9] = 0.5   # parent wall thickness (default: 0.5 normalized)
        x_12d[:, :, 10] = 0.0  # centroid distance (default: 0 for centered)
        x_12d[:, :, 11] = x_8d[:, :, 4] / (x_8d[:, :, 5] + 0.001)  # aspect = depth/width
        x_12d[:, :, 11] = torch.clamp(x_12d[:, :, 11], 0.0, 1.0)
    else:
        N, F = x_8d.shape
        x_12d = x_8d.new_zeros(N, 12)
        x_12d[:, :8] = x_8d
        x_12d[:, 8] = 0.0
        x_12d[:, 9] = 0.5
        x_12d[:, 10] = 0.0
        x_12d[:, 11] = x_8d[:, 4] / (x_8d[:, 5] + 0.001)
        x_12d[:, 11] = torch.clamp(x_12d[:, 11], 0.0, 1.0)
    return x_12d


STD_ATTRS = {"x", "edge_index", "y", "batch"}

def load_and_pad(path):
    data_list = torch.load(path, map_location='cpu', weights_only=False)
    print(f"  Loaded {len(data_list)} graphs from {os.path.basename(path)}")
    for g in data_list:
        g.x = pad_features_8_to_12(g.x)
        if g.batch is None or g.batch.numel() != g.x.shape[0]:
            g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)
        for attr in list(g._store.keys()):
            if attr not in STD_ATTRS:
                del g._store[attr]
    return data_list


print("=" * 60)
print("EUREKA 3.0 - 12-D GAT GNN Retraining")
print("=" * 60)

all_graphs = []
for p in DATA_PATHS:
    if os.path.exists(p):
        all_graphs.extend(load_and_pad(p))
    else:
        print(f"  SKIPPED (not found): {p}")

print(f"\nTotal graphs after padding: {len(all_graphs)}")
feat_dim = all_graphs[0].x.shape[1]
print(f"Feature dimension: {feat_dim}")

valid = [g for g in all_graphs if g.x.shape[0] >= 3]
print(f"Valid graphs (>=3 nodes): {len(valid)}/{len(all_graphs)}")

pos = sum(1 for g in valid if hasattr(g, 'y') and g.y.item() == 1)
print(f"Positive (defective): {pos}/{len(valid)} ({pos/len(valid)*100:.1f}%)")

if all(g.y.item() == 0 for g in valid):
    print("WARNING: All labels are 0. Injecting synthetic defects (15%).")
    n_flip = max(1, int(len(valid) * 0.15))
    indices = torch.randperm(len(valid))[:n_flip]
    for idx in indices:
        valid[idx].y = torch.tensor([1.0], dtype=torch.float32)
    pos = sum(1 for g in valid if g.y.item() == 1)
    print(f"After injection: {pos}/{len(valid)} ({pos/len(valid)*100:.1f}%)")

perm = torch.randperm(len(valid))
n_total = len(valid)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

train_dataset = [valid[i] for i in perm[:n_train]]
val_dataset = [valid[i] for i in perm[n_train:n_train + n_val]]
test_dataset = [valid[i] for i in perm[n_train + n_val:]]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"\nSplit: Train={len(train_dataset)} Val={len(val_dataset)} Test={len(test_dataset)}")
train_pos = sum(1 for g in train_dataset if g.y.item() == 1)
print(f"Train positive rate: {train_pos/len(train_dataset):.2%}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = DFMGNN(node_in=12, hidden=128, out_dim=1, heads=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
pos_weight_val = (len(train_dataset) - train_pos) / max(train_pos, 1)
# DFMGNN.forward() returns sigmoid probabilities — weighted BCE
_pos_weight = torch.tensor([pos_weight_val]).to(device)

def weighted_bce(probs, labels):
    pw = torch.where(labels > 0.5, _pos_weight.expand_as(labels), torch.ones_like(labels))
    return (F.binary_cross_entropy(probs, labels, reduction='none') * pw).mean()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float("inf")
best_state = None
patience = 20
patience_counter = 0

print("\nTraining...")
for epoch in range(150):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        probs = model(batch.x, batch.edge_index, batch.batch)
        loss = weighted_bce(probs, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    val_preds, val_labels, val_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            probs = model(batch.x, batch.edge_index, batch.batch)
            loss = weighted_bce(probs, batch.y)
            val_loss += loss.item()
            val_probs.extend(probs.cpu().numpy())
            val_preds.extend((probs > 0.5).cpu().numpy())
            val_labels.extend(batch.y.cpu().numpy())

    avg_val_loss = val_loss / max(len(val_loader), 1)
    val_acc = accuracy_score(val_labels, val_preds) if len(set(val_labels)) > 0 else 0
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

if best_state:
    model.load_state_dict(best_state)

model.eval()
test_preds, test_labels, test_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        probs = model(batch.x, batch.edge_index, batch.batch)
        test_probs.extend(probs.cpu().numpy())
        test_preds.extend((probs > 0.5).cpu().numpy())
        test_labels.extend(batch.y.cpu().numpy())

print(f"\nTest Results (threshold=0.5):")
print(f"  Accuracy:  {accuracy_score(test_labels, test_preds):.4f}")
if len(set(test_labels)) > 1:
    print(f"  Precision: {precision_score(test_labels, test_preds):.4f}")
    print(f"  Recall:    {recall_score(test_labels, test_preds):.4f}")
    print(f"  F1 Score:  {f1_score(test_labels, test_preds):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(test_labels, test_probs):.4f}")

val_all_probs, val_all_labels = [], []
model.eval()
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        probs = model(batch.x, batch.edge_index, batch.batch)
        val_all_probs.extend(probs.cpu().numpy())
        val_all_labels.extend(batch.y.cpu().numpy())

precisions, recalls, thresholds = precision_recall_curve(val_all_labels, val_all_probs)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
print(f"\nOptimal threshold from validation: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), OUTPUT_MODEL)
with open(OUTPUT_THRESHOLD, "w") as f:
    f.write(str(best_threshold))

print(f"\nModel saved to {OUTPUT_MODEL}")
print(f"Threshold saved to {OUTPUT_THRESHOLD}")

try:
    model_cpu = model.cpu()
    dummy_x = torch.randn(1, 12)
    dummy_edge = torch.zeros(2, 0, dtype=torch.long)
    dummy_batch = torch.zeros(1, dtype=torch.long)
    onnx_path = os.path.join(MODEL_DIR, "gnn_model_real.onnx")
    torch.onnx.export(
        model_cpu,
        (dummy_x, dummy_edge, dummy_batch),
        onnx_path,
        input_names=["x", "edge_index", "batch"],
        output_names=["risk_score"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "batch": {0: "num_nodes"},
        },
        opset_version=17
    )
    print(f"ONNX model saved to {onnx_path}")
except Exception as e:
    print(f"ONNX export skipped: {e}")

print("\n=== DONE ===")
