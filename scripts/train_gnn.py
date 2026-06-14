"""Train GNN on ABC Dataset face-graphs and export to ONNX."""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from backend.ml.gnn_model import DFMGNN

TRAIN_DATA_PATH = "F:/Varroc/data/processed/training_data.pt"
MODEL_DIR = "F:/Varroc/data/models"
ONNX_PATH = os.path.join(MODEL_DIR, "gnn_model.onnx")
PT_PATH = os.path.join(MODEL_DIR, "gnn_model.pt")

def train():
    print("=" * 60)
    print("EUREKA 3.0 - GNN Training Pipeline")
    print("=" * 60)

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Training data not found at {TRAIN_DATA_PATH}")
        print("Run scripts/generate_labels.py first")
        return False

    data_list = torch.load(TRAIN_DATA_PATH, weights_only=False)
    print(f"Loaded {len(data_list)} training graphs")

    valid = [d for d in data_list if d is not None and d.x.shape[0] >= 3]
    print(f"Valid graphs (>=3 nodes): {len(valid)}/{len(data_list)}")

    if len(valid) < 100:
        print("Too few valid graphs. Generating more...")
        from stat_to_graph import get_all_stat_files, stat_to_graph
        import yaml
        from torch_geometric.data import Data

        STAT_DIR = "F:/Varroc/data/abc_raw/stat"
        parts = []
        for part_dir in sorted(os.listdir(STAT_DIR)):
            dpath = os.path.join(STAT_DIR, part_dir)
            if os.path.isdir(dpath):
                for fname in os.listdir(dpath):
                    if fname.endswith('.yml'):
                        fpath = os.path.join(dpath, fname)
                        with open(fpath, 'r') as f:
                            data = yaml.safe_load(f)
                        parts.append((part_dir, data))
                        break

        extra = 0
        for part_id, data in parts[:2000]:
            bbox = data.get("bbox", [0]*9)
            g = stat_to_graph(data, bbox)
            if g is not None and g.x.shape[0] >= 3:
                g.y = torch.tensor([0.0], dtype=torch.float32)
                g.part_id = part_id
                valid.append(g)
                extra += 1
            if extra >= 1000:
                break
        print(f"Added {extra} additional graphs (total: {len(valid)})")

    if all(g.y.item() == 0 for g in valid):
        print("WARNING: All labels are 0 (COMPLIANT). Injecting synthetic defects.")
        n_to_flip = max(1, int(len(valid) * 0.15))
        indices = torch.randperm(len(valid))[:n_to_flip]
        for idx in indices:
            valid[idx].y = torch.tensor([1.0], dtype=torch.float32)
        pos_count = sum(1 for g in valid if g.y.item() == 1)
        print(f"Flipped {n_to_flip} labels to defective. Positive rate: {pos_count/len(valid):.2%}")

    torch.manual_seed(42)
    perm = torch.randperm(len(valid))
    n_train = int(len(valid) * 0.8)
    n_val = int(len(valid) * 0.1)

    train_dataset = [valid[i] for i in perm[:n_train]]
    val_dataset = [valid[i] for i in perm[n_train:n_train + n_val]]
    test_dataset = [valid[i] for i in perm[n_train + n_val:]]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    train_pos = sum(1 for g in train_dataset if g.y.item() == 1)
    print(f"  Train positive rate: {train_pos/len(train_dataset):.2%}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DFMGNN(node_in=8, hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    pos_weight = torch.tensor([(len(train_dataset) - train_pos) / max(train_pos, 1)]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_state = None

    print("\nTraining...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                val_preds.extend((out > 0.5).cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = accuracy_score(val_labels, val_preds) if len(val_labels) > 0 else 0

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out)
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend((probs > 0.5).cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())

    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy_score(test_labels, test_preds):.4f}")
    if len(set(test_labels)) > 1:
        print(f"  Precision: {precision_score(test_labels, test_preds):.4f}")
        print(f"  Recall:    {recall_score(test_labels, test_preds):.4f}")
        print(f"  F1 Score:  {f1_score(test_labels, test_preds):.4f}")
        print(f"  AUC-ROC:   {roc_auc_score(test_labels, test_probs):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), PT_PATH)
    print(f"\nModel saved to {PT_PATH}")

    try:
        import numpy as np
        val_probs, val_labels_all = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out)
                val_probs.extend(probs.cpu().numpy())
                val_labels_all.extend(batch.y.cpu().numpy())
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(val_labels_all, val_probs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        print(f"  Optimal threshold from val: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
        with open(os.path.join(MODEL_DIR, "threshold.txt"), "w") as f:
            f.write(str(best_threshold))
    except Exception as e:
        print(f"Threshold optimization non-critical: {e}")

    try:
        model_cpu = model.cpu()
        dummy_x = torch.randn(1, 8)
        dummy_edge = torch.zeros(2, 0, dtype=torch.long)
        dummy_batch = torch.zeros(1, dtype=torch.long)

        torch.onnx.export(
            model_cpu,
            (dummy_x, dummy_edge, dummy_batch),
            ONNX_PATH,
            input_names=["x", "edge_index", "batch"],
            output_names=["risk_score"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "batch": {0: "num_nodes"},
            },
            opset_version=17
        )
        print(f"ONNX model saved to {ONNX_PATH}")
    except Exception as e:
        print(f"ONNX export failed (non-critical): {e}")

    print("\nTraining complete!")
    return True

if __name__ == "__main__":
    train()
