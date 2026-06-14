"""Retrain GNN on real feat data (small files only) and export ONNX."""
import os, sys, yaml, time, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool
import torch.nn.functional as F

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly

FEAT_DIR = "F:/Varroc/data/abc_raw/feat"
OUTPUT_DIR = "F:/Varroc/data/processed"
MODEL_DIR = "F:/Varroc/data/models"
MAX_FILE_MB = 2
MAX_GRAPHS = 500

TYPE_MAP = {"Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3, "Torus": 4, "BSpline": 5}
FT_MAP = {"Plane": FaceType.PLANE, "Cylinder": FaceType.CYLINDER, "Sphere": FaceType.SPHERE,
          "Cone": FaceType.CONE, "Torus": FaceType.TORUS, "BSpline": FaceType.BSPLINE}

yaml_loader = yaml.CLoader if hasattr(yaml, 'CLoader') else yaml.Loader

def estimate_area(coefficients, vert_count, bbox_dims):
    if vert_count < 4:
        return 100.0
    x, y, z = bbox_dims
    return max((x * y * z) ** 0.667 * (vert_count / 1000.0), 10.0)

def estimate_curvature(face_type, radius, coefficients):
    if face_type == "Plane":
        return 0.0, 0.0
    if radius and radius > 0:
        return 1.0 / radius, 1.0 / (radius * 2)
    if face_type == "Sphere":
        return 1.0, 1.0
    if face_type == "Cylinder":
        return 1.0, 0.0
    return 0.5, 0.5

# Step 1: Scan + Collect small files
print("=== Step 1: Scanning feat files (under {}MB) ===".format(MAX_FILE_MB))
t0 = time.time()
candidates = []
for part_dir in sorted(os.listdir(FEAT_DIR)):
    dp = os.path.join(FEAT_DIR, part_dir)
    if not os.path.isdir(dp):
        continue
    for fname in os.listdir(dp):
        if not fname.endswith('.yml'):
            continue
        fp = os.path.join(dp, fname)
        sz = os.path.getsize(fp)
        if sz < MAX_FILE_MB * 1024 * 1024:
            candidates.append((part_dir, fp, sz))
print("Found {} candidates under {}MB in {:.1f}s".format(len(candidates), MAX_FILE_MB, time.time()-t0))

# Step 2: Parse YAML + Build graphs
print("\n=== Step 2: Parsing YAML + building graphs ===")
t1 = time.time()
graphs = []
labels_list = []
skipped = 0
MAX_SURF = 200

for idx, (part_id, fp, sz) in enumerate(candidates):
    if len(graphs) >= MAX_GRAPHS:
        break
    if idx % 100 == 0 and idx > 0:
        elapsed = time.time() - t1
        rate = idx / elapsed
        remaining = (min(len(candidates), MAX_GRAPHS + skipped) - idx) / rate
        print("  [{}/{}] {} graphs built, {:.0f}s elapsed, ETA {:.0f}s".format(
            idx, min(len(candidates), MAX_GRAPHS + skipped), len(graphs), elapsed, remaining))

    try:
        with open(fp, 'rb') as f:
            data = yaml.load(f, Loader=yaml_loader)
    except:
        skipped += 1
        continue

    surfs = data.get("surfaces", [])
    curves = data.get("curves", [])
    n = len(surfs)
    if n < 3 or n > MAX_SURF:
        skipped += 1
        continue

    # Build node features and adjacency
    x_list = []
    faces_meta = []
    vert_to_surfaces = {}
    bbox_dims = (100.0, 100.0, 100.0)

    for i, s in enumerate(surfs):
        st = s.get("type", "Other")
        radius = s.get("radius") or 0.0
        coeffs = s.get("coefficients", [])
        verts = s.get("vert_indices", [])
        location = s.get("location", [0, 0, 0])

        ft_idx = TYPE_MAP.get(st, 6)
        vert_count = len(verts)
        area = estimate_area(coeffs, vert_count, bbox_dims)
        thickness = 3.0
        draft = 2.5 if st == "Plane" else 3.0
        radius_val = radius if radius and radius > 0 else 2.0
        depth = 5.0
        width = math.sqrt(area) if area > 0 else 5.0
        curv_min, curv_max = estimate_curvature(st, radius, coeffs)

        x_list.append([
            ft_idx / 7.0,
            min(thickness / 5.0, 1.0),
            min(float(radius_val) / 10.0, 1.0),
            min(float(area) / 10000.0, 1.0),
            min(depth / 50.0, 1.0),
            min(width / 50.0, 1.0),
            min(abs(curv_min) / 10.0, 1.0),
            min(abs(curv_max) / 10.0, 1.0)
        ])

        ft = FT_MAP.get(st, FaceType.OTHER)
        faces_meta.append(FaceGeometry(
            face_id="{}_face_{:04d}".format(part_id, i),
            face_type=ft, area_mm2=float(area), thickness_mm=thickness,
            draft_angle_deg=draft, radius_mm=float(radius_val),
            depth_mm=depth, width_mm=width,
            sw_feature_name="Surface_{}".format(i), sw_feature_type=st,
            curvature_min=float(curv_min), curvature_max=float(curv_max),
            centroid_x=float(location[0]), centroid_y=float(location[1]), centroid_z=float(location[2])
        ))

        for v in verts:
            if v not in vert_to_surfaces:
                vert_to_surfaces[v] = []
            vert_to_surfaces[v].append(i)

    x = torch.tensor(x_list, dtype=torch.float32)

    adj_set = set()
    for v, s_list in vert_to_surfaces.items():
        unique_s = list(set(s_list))
        for ii in range(len(unique_s)):
            for jj in range(ii + 1, len(unique_s)):
                a, b = unique_s[ii], unique_s[jj]
                adj_set.add((a, b) if a < b else (b, a))

    edge_list = []
    for a, b in adj_set:
        edge_list.append([a, b])
        edge_list.append([b, a])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, part_id=part_id, num_surfaces=n)

    # Run rules for label
    bb = [0, 0, 0, 100, 100, 100]
    part = PartMetadata(
        filename="abc_{}".format(part_id),
        bounding_box_mm="100 x 100 x 100",
        face_count=len(faces_meta),
        faces=faces_meta
    )
    violations = engine.validate(part)
    has_real = any(v.severity.value in ("CRITICAL", "WARNING") for v in violations)
    is_def = int(has_real)
    graph.y = torch.tensor([float(is_def)], dtype=torch.float32)

    graph.batch = torch.zeros(n, dtype=torch.long)
    graphs.append(graph)
    labels_list.append({
        "part_id": part_id,
        "face_count": n,
        "violation_count": len(violations),
        "is_defective": is_def,
    })

t2 = time.time()
defective = sum(1 for g in graphs if g.y.item() == 1)
print("\nBuilt {} real face-graphs in {:.0f}s".format(len(graphs), t2 - t1))
print("Defective: {}/{} ({:.1f}%)".format(defective, len(graphs), defective/len(graphs)*100))
print("Skipped: {}".format(skipped))

# Step 3: Train GNN
print("\n=== Step 3: Training GNN ===")
torch.manual_seed(42)
np.random.seed(42)

n_graphs = len(graphs)
n_train = int(n_graphs * 0.7)
n_val = int(n_graphs * 0.15)
idx_perm = np.random.permutation(n_graphs)
train_idx = idx_perm[:n_train]
val_idx = idx_perm[n_train:n_train+n_val]
test_idx = idx_perm[n_train+n_val:]

train_data = [graphs[i] for i in train_idx]
val_data = [graphs[i] for i in val_idx]
test_data = [graphs[i] for i in test_idx]

print("Train: {}, Val: {}, Test: {}".format(len(train_data), len(val_data), len(test_data)))

class SimpleGNN(torch.nn.Module):
    def __init__(self, node_in=8, hidden=64, out_dim=1):
        super().__init__()
        self.conv1 = GraphConv(node_in, hidden)
        self.conv2 = GraphConv(hidden, 32)
        self.lin1 = torch.nn.Linear(32, 16)
        self.lin2 = torch.nn.Linear(16, out_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.sigmoid(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = SimpleGNN(8, 64, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = torch.nn.BCELoss()

def train_epoch(dataset):
    model.train()
    total_loss = 0
    for g in dataset:
        g = g.to(device)
        optimizer.zero_grad()
        out = model(g.x, g.edge_index, g.batch)
        loss = loss_fn(out, g.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)

@torch.no_grad()
def evaluate(dataset):
    model.eval()
    preds, labels = [], []
    for g in dataset:
        g = g.to(device)
        out = model(g.x, g.edge_index, g.batch)
        preds.append(out.item())
        labels.append(g.y.item())
    preds = np.array(preds)
    labels = np.array(labels)
    from sklearn.metrics import roc_auc_score, f1_score
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    # Find optimal threshold
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(labels, (preds >= th).astype(int))
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return auc, best_f1, best_th

best_val_auc = 0
patience = 10
wait = 0

for epoch in range(200):
    loss = train_epoch(train_data)
    val_auc, val_f1, val_th = evaluate(val_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_threshold = val_th
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gnn_model_real.pt"))
        wait = 0
    else:
        wait += 1
    if (epoch+1) % 20 == 0 or epoch == 0:
        print("Epoch {:3d} | Loss: {:.4f} | Val AUC: {:.4f} F1: {:.4f} th: {:.3f}".format(
            epoch+1, loss, val_auc, val_f1, val_th))
    if wait >= patience:
        print("Early stopping at epoch", epoch+1)
        break

# Load best model
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gnn_model_real.pt")))
test_auc, test_f1, _ = evaluate(test_data)
print("\n=== Test Results ===")
print("Test AUC: {:.4f}, F1: {:.4f}".format(test_auc, test_f1))
print("Best val AUC: {:.4f}, threshold: {:.4f}".format(best_val_auc, best_threshold))

with open(os.path.join(MODEL_DIR, "threshold_real.txt"), "w") as f:
    f.write(str(best_threshold))

# Step 4: ONNX export (on CPU for compatibility)
print("\n=== Step 4: ONNX Export ===")
try:
    model_cpu = SimpleGNN(8, 64, 1)
    model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    model_cpu.eval()
    dummy_x = torch.randn(10, 8)
    dummy_edge = torch.randint(0, 10, (2, 20))
    dummy_batch = torch.zeros(10, dtype=torch.long)

    torch.onnx.export(
        model_cpu,
        (dummy_x, dummy_edge, dummy_batch),
        os.path.join(MODEL_DIR, "gnn_model_real.onnx"),
        input_names=["x", "edge_index", "batch"],
        output_names=["probability"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "batch": {0: "num_nodes"},
        },
        opset_version=17
    )
    print("ONNX exported successfully")
except Exception as e:
    print("ONNX export failed:", e)

# Save dataset
pt_path = os.path.join(OUTPUT_DIR, "real_training_data.pt")
torch.save(graphs, pt_path)
print("Dataset saved: {} graphs -> {}".format(len(graphs), pt_path))

csv_path = os.path.join(OUTPUT_DIR, "real_labels.csv")
with open(csv_path, "w", newline="") as f:
    if labels_list:
        w = csv.DictWriter(f, fieldnames=list(labels_list[0].keys()))
        w.writeheader()
        w.writerows(labels_list)
print("Labels saved to", csv_path)

print("\n=== DONE ===")
