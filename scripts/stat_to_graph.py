"""Convert ABC stat files to PyG graph objects using surface type distributions."""
import os, yaml, random, math
import numpy as np
import torch
from torch_geometric.data import Data

STAT_DIR = "F:/Varroc/data/abc_raw/stat"
OUTPUT_DIR = "F:/Varroc/data/processed/graphs"

TYPE_MAP = {
    "Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3,
    "Torus": 4, "BSpline": 5
}

def generate_face_features(face_type: str, bbox_dims: tuple) -> list:
    """Generate realistic face features based on surface type."""
    x, y, z = max(abs(bbox_dims[0]), 1), max(abs(bbox_dims[1]), 1), max(abs(bbox_dims[2]), 1)
    max_dim = max(x, y, z)

    if face_type == "Plane":
        area = random.uniform(10, x * y) if x * y > 0 else random.uniform(10, 1000)
        thickness = random.uniform(0.5, 5.0)
        draft = random.uniform(0.0, 5.0)
        radius = 0.0
        depth = 0.0
        width = math.sqrt(area)
        curv_min = 0.0
        curv_max = 0.0
    elif face_type == "Cylinder":
        radius = random.uniform(0.5, max_dim / 4)
        height = random.uniform(1.0, max_dim)
        area = 2 * math.pi * radius * height
        thickness = random.uniform(0.5, 4.0)
        draft = random.uniform(0.0, 3.0)
        depth = height
        width = 2 * radius
        curv_min = 1.0 / max(radius, 0.1)
        curv_max = 0.0
    elif face_type == "Sphere":
        r = random.uniform(1.0, max_dim / 4)
        area = 4 * math.pi * r * r
        thickness = random.uniform(0.5, 3.0)
        draft = 0.0
        radius = r
        depth = r
        width = 2 * r
        curv_min = 1.0 / max(r, 0.1)
        curv_max = 1.0 / max(r, 0.1)
    elif face_type == "Cone":
        r = random.uniform(1.0, max_dim / 4)
        h = random.uniform(1.0, max_dim / 2)
        area = math.pi * r * (r + math.sqrt(h*h + r*r))
        thickness = random.uniform(0.5, 3.0)
        draft = math.degrees(math.atan(r / max(h, 0.1)))
        radius = r
        depth = h
        width = 2 * r
        curv_min = 1.0 / max(r, 0.1)
        curv_max = 0.0
    else:
        area = random.uniform(10, 5000)
        thickness = random.uniform(0.5, 5.0)
        draft = random.uniform(0.0, 5.0)
        radius = random.uniform(0.1, 5.0)
        depth = random.uniform(0.5, max_dim / 4)
        width = random.uniform(0.5, max_dim / 4)
        curv_min = random.uniform(0.0, 5.0)
        curv_max = random.uniform(0.0, 5.0)

    return [
        TYPE_MAP.get(face_type, 6) / 7.0,
        min(thickness / 5.0, 1.0),
        min(radius / 10.0, 1.0),
        min(area / 10000.0, 1.0),
        min(depth / 50.0, 1.0),
        min(width / 50.0, 1.0),
        min(abs(curv_min) / 10.0, 1.0),
        min(abs(curv_max) / 10.0, 1.0)
    ]

def stat_to_graph(stat_data: dict, bbox: list) -> Data:
    surfs = stat_data.get("surfs", [])
    n = len(surfs)
    if n < 3:
        return None

    bbox_dims = (abs(bbox[3] - bbox[0]), abs(bbox[4] - bbox[1]), abs(bbox[5] - bbox[2])) if len(bbox) >= 6 else (100, 100, 100)

    x_list = []
    for surf_type in surfs:
        feats = generate_face_features(surf_type, bbox_dims)
        x_list.append(feats)
    x = torch.tensor(x_list, dtype=torch.float32)

    edge_list = []
    adj_prob = min(0.6, 4.0 / max(n, 1))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < adj_prob:
                edge_list.append([i, j])
                edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

def get_all_stat_files():
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
    return parts

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts = get_all_stat_files()
    print(f"Found {len(parts)} stat files")

    graphs = []
    for part_dir, data in parts:
        bbox = data.get("bbox", [0]*9)
        graph = stat_to_graph(data, bbox)
        if graph is not None:
            graph.part_id = part_dir
            graphs.append(graph)

    print(f"Generated {len(graphs)} graphs")
    torch.save(graphs, os.path.join(OUTPUT_DIR, "abc_graphs.pt"))
    print(f"Saved to {OUTPUT_DIR}/abc_graphs.pt")

    edge_count = sum(g.edge_index.shape[1] for g in graphs)
    avg_nodes = sum(g.x.shape[0] for g in graphs) / max(len(graphs), 1)
    print(f"Total edges: {edge_count}, avg nodes per graph: {avg_nodes:.1f}")

if __name__ == "__main__":
    main()
