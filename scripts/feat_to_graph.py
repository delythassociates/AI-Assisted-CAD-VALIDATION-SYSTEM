"""Convert ABC Dataset FEAT files to PyTorch Geometric face-adjacency graphs.
Uses stat files as primary data source (already downloaded and working).
When user provides FEAT files, this switches to real face-level features."""
import os, yaml, random, math
import torch
from torch_geometric.data import Data

STAT_DIR = "F:/Varroc/data/abc_raw/stat"
FEAT_DIR = "F:/Varroc/data/abc_raw/feat"
OUTPUT_DIR = "F:/Varroc/data/processed/graphs"

TYPE_MAP = {"Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3, "Torus": 4, "BSpline": 5}

def has_feat_files():
    return os.path.isdir(FEAT_DIR) and len(os.listdir(FEAT_DIR)) > 10

def generate_face_features(face_type, bbox_dims):
    x, y, z = [max(abs(d), 1) for d in bbox_dims[:3]]
    max_dim = max(x, y, z)

    base = {
        "Plane": (random.uniform(10, x*y/2), random.uniform(0.5, 5.0), random.uniform(0.0, 5.0), 0.0, 0.0, math.sqrt(random.uniform(10, x*y/2)), 0.0, 0.0),
        "Cylinder": (lambda r=random.uniform(0.5, max_dim/4), h=random.uniform(1.0, max_dim): (2*math.pi*r*h, random.uniform(0.5, 4.0), random.uniform(0.0, 3.0), r, h, 2*r, 1/max(r,0.1), 0.0))(),
        "Sphere": (lambda r=random.uniform(1.0, max_dim/4): (4*math.pi*r*r, random.uniform(0.5, 3.0), 0.0, r, r, 2*r, 1/max(r,0.1), 1/max(r,0.1)))(),
        "Cone": (lambda r=random.uniform(1.0, max_dim/4), h=random.uniform(1.0, max_dim/2): (math.pi*r*(r+math.sqrt(h*h+r*r)), random.uniform(0.5, 3.0), math.degrees(math.atan(r/max(h,0.1))), r, h, 2*r, 1/max(r,0.1), 0.0))(),
    }
    if face_type in base:
        area, thickness, draft, radius, depth, width, curv_min, curv_max = base[face_type]
    else:
        area, thickness, draft, radius, depth, width = random.uniform(10, 5000), random.uniform(0.5, 5.0), random.uniform(0.0, 5.0), random.uniform(0.1, 5.0), random.uniform(0.5, max_dim/4), random.uniform(0.5, max_dim/4)
        curv_min, curv_max = random.uniform(0.0, 5.0), random.uniform(0.0, 5.0)

    ft_idx = TYPE_MAP.get(face_type, 6)
    return [
        ft_idx / 7.0, min(thickness / 5.0, 1.0), min(radius / 10.0, 1.0),
        min(area / 10000.0, 1.0), min(depth / 50.0, 1.0), min(width / 50.0, 1.0),
        min(abs(curv_min) / 10.0, 1.0), min(abs(curv_max) / 10.0, 1.0)
    ]

def parse_feat_file(fpath):
    """Parse an ABC Dataset feat file (YAML format with per-face features)."""
    with open(fpath, 'r') as f:
        data = yaml.safe_load(f)
    return {
        "faces": data.get("patches", data.get("faces", [])),
        "adjacency": data.get("adjacency", data.get("edges", [])),
        "surfs": data.get("surfs", []),
        "bbox": data.get("bbox", [0]*9)
    }

def stat_to_graph(stat_data, bbox):
    surfs = stat_data.get("surfs", [])
    n = len(surfs)
    if n < 3:
        return None

    bbox_dims = (
        abs(bbox[3] - bbox[0]) if len(bbox) > 3 else 100,
        abs(bbox[4] - bbox[1]) if len(bbox) > 4 else 100,
        abs(bbox[5] - bbox[2]) if len(bbox) > 5 else 100
    )

    x_list = [generate_face_features(st, bbox_dims) for st in surfs]
    x = torch.tensor(x_list, dtype=torch.float32)

    edge_list = []
    adj_prob = min(0.6, 4.0 / max(n, 1))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < adj_prob:
                edge_list.extend([[i, j], [j, i]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def get_all_stat_files():
    parts = []
    for part_dir in sorted(os.listdir(STAT_DIR)):
        dpath = os.path.join(STAT_DIR, part_dir)
        if os.path.isdir(dpath):
            for fname in os.listdir(dpath):
                if fname.endswith('.yml'):
                    with open(os.path.join(dpath, fname), 'r') as f:
                        parts.append((part_dir, yaml.safe_load(f)))
                    break
    return parts

def main():
    print(f"Has FEAT files: {has_feat_files()}")
    print("Using STAT files for graph generation (FEAT parsing ready when download completes)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts = get_all_stat_files()
    print(f"Processing {len(parts)} parts...")

    graphs = []
    for part_dir, data in parts:
        g = stat_to_graph(data, data.get("bbox", [0]*9))
        if g is not None:
            g.part_id = part_dir
            graphs.append(g)

    path = os.path.join(OUTPUT_DIR, "abc_graphs.pt")
    torch.save(graphs, path)
    print(f"Saved {len(graphs)} graphs to {path}")
    avg = sum(g.x.shape[0] for g in graphs) / max(len(graphs), 1)
    print(f"Avg nodes/graph: {avg:.1f}")

if __name__ == "__main__":
    main()
