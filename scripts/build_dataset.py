"""Complete dataset pipeline: stat files → graphs → labels → training data."""
import os, sys, yaml, random, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch_geometric.data import Data

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly

STAT_DIR = "F:/Varroc/data/abc_raw/stat"
OUTPUT_DIR = "F:/Varroc/data/processed"
MODEL_DIR = "F:/Varroc/data/models"

TYPE_MAP = {"Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3, "Torus": 4}
FT_MAP = {"Plane": FaceType.PLANE, "Cylinder": FaceType.CYLINDER, "Sphere": FaceType.SPHERE,
          "Cone": FaceType.CONE, "Torus": FaceType.TORUS}

yaml_loader = yaml.CLoader if hasattr(yaml, 'CLoader') else yaml.Loader

DEFECT_PROB = 0.18
RNG = random.Random(42)

def gen_face_feats(face_type, bbox_dims, is_defective_part=False):
    x, y, z = [max(abs(d), 1) for d in bbox_dims[:3]]
    mx = max(x, y, z)

    compliant = not is_defective_part
    if face_type == "Plane":
        area = RNG.uniform(100, max(x*y/2, 500))
        t = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.3, 0.7)
        d = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.1, 0.8)
        r, dp, w = 3.0, 0.0, math.sqrt(area)
        cm, cx = 0.0, 0.0
    elif face_type == "Cylinder":
        rad = RNG.uniform(3.0, mx/5)
        h = RNG.uniform(5.0, mx/2)
        area = 2 * math.pi * rad * h
        t = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.5, 1.5)
        d = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.1, 1.0)
        r, dp, w = rad, h, 2*rad
        cm, cx = 1/max(rad, 0.1), 0.0
    elif face_type == "Sphere":
        rad = RNG.uniform(3.0, mx/5)
        area = 4 * math.pi * rad * rad
        t = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.5, 1.5)
        d = 0.0
        r, dp, w = rad, rad, 2*rad
        cm = cx = 1/max(rad, 0.1)
    elif face_type == "Cone":
        rad = RNG.uniform(3.0, mx/5)
        h = RNG.uniform(5.0, mx/3)
        area = math.pi * rad * (rad + math.sqrt(h*h + rad*rad))
        t = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.5, 1.5)
        d = math.degrees(math.atan(rad/max(h, 0.1)))
        r, dp, w = rad, h, 2*rad
        cm, cx = 1/max(rad, 0.1), 0.0
    else:
        area = RNG.uniform(100, 5000)
        t = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.3, 0.7)
        d = RNG.uniform(3.0, 5.0) if compliant else RNG.uniform(0.1, 0.8)
        r = RNG.uniform(2.0, 5.0) if compliant else RNG.uniform(0.1, 0.4)
        dp = RNG.uniform(1.0, mx/5) if compliant else RNG.uniform(10.0, mx/2)
        w = RNG.uniform(1.0, mx/5) if compliant else RNG.uniform(0.5, 2.0)
        cm, cx = abs(RNG.gauss(0, 1)), abs(RNG.gauss(0, 1))

    ft_idx = TYPE_MAP.get(face_type, 6)
    return [
        ft_idx / 7.0, min(t/5.0, 1.0), min(r/10.0, 1.0), min(area/10000.0, 1.0),
        min(dp/50.0, 1.0), min(w/50.0, 1.0), min(abs(cm)/10.0, 1.0), min(abs(cx)/10.0, 1.0)
    ], area, t, d, r

def build_part_and_graph(part_id, stat_data):
    surfs = stat_data.get("surfs", [])
    bbox = stat_data.get("bbox", [0]*9)
    n = len(surfs)
    if n < 3:
        return None, None

    bbox_dims = (
        abs(bbox[3]-bbox[0]) if len(bbox) > 3 else 100,
        abs(bbox[4]-bbox[1]) if len(bbox) > 4 else 100,
        abs(bbox[5]-bbox[2]) if len(bbox) > 5 else 100
    )
    bbox_str = f"{bbox_dims[0]:.1f} x {bbox_dims[1]:.1f} x {bbox_dims[2]:.1f}"

    is_defective_part = RNG.random() < DEFECT_PROB
    defective_faces = set()
    if is_defective_part:
        num_defective = max(1, int(n * RNG.uniform(0.1, 0.3)))
        defective_faces = set(RNG.sample(range(n), min(num_defective, n)))

    faces = []
    x_list = []
    for i, st in enumerate(surfs):
        is_def = i in defective_faces
        feats, area, thick, draft, rad = gen_face_feats(st, bbox_dims, is_defective_part=is_def)
        x_list.append(feats)
        ft = FT_MAP.get(st, FaceType.OTHER)
        faces.append(FaceGeometry(
            face_id=f"{part_id}_face_{i:04d}", face_type=ft,
            area_mm2=area, thickness_mm=thick, draft_angle_deg=draft, radius_mm=rad,
            depth_mm=0.0, width_mm=0.0, sw_feature_name=f"Surface_{i}", sw_feature_type=st
        ))

    x = torch.tensor(x_list, dtype=torch.float32)
    adj_prob = min(0.6, 4.0 / max(n, 1))
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < adj_prob:
                edge_list.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, part_id=part_id)

    part = PartMetadata(
        filename=f"abc_{part_id}", bounding_box_mm=bbox_str,
        face_count=n, faces=faces
    )

    return part, graph

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    t0 = time.time()
    stat_parts = []
    for part_dir in sorted(os.listdir(STAT_DIR)):
        dpath = os.path.join(STAT_DIR, part_dir)
        if os.path.isdir(dpath):
            for fname in os.listdir(dpath):
                if fname.endswith('.yml'):
                    with open(os.path.join(dpath, fname), 'r') as f:
                        stat_parts.append((part_dir, yaml.load(f, Loader=yaml_loader)))
                    break
    t1 = time.time()
    print(f"Loaded {len(stat_parts)} stat files in {t1-t0:.1f}s")

    graphs = []
    labels_list = []
    train_data = []

    for idx, (part_id, sd) in enumerate(stat_parts):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(stat_parts)}... ({time.time()-t1:.0f}s)")

        part, graph = build_part_and_graph(part_id, sd)
        if part is None:
            continue

        violations = engine.validate(part)
        has_real_violation = any(v.severity.value in ("CRITICAL", "WARNING") for v in violations)
        is_def = int(has_real_violation)

        if graph is not None:
            graph.y = torch.tensor([float(is_def)], dtype=torch.float32)
            train_data.append(graph)
            graphs.append(graph)

        labels_list.append({
            "part_id": part_id, "face_count": part.face_count,
            "violation_count": len(violations),
            "is_defective": is_def,
            "volume": sd.get("volume", 0),
            "surface_area": sd.get("surface", 0),
        })

    t2 = time.time()
    print(f"\nGenerated {len(graphs)} graphs in {t2-t1:.0f}s")
    
    defective = sum(1 for g in train_data if g.y.item() == 1)
    print(f"Defective: {defective}/{len(train_data)} ({defective/max(len(train_data),1)*100:.1f}%)")
    avg_nodes = sum(g.x.shape[0] for g in train_data) / max(len(train_data), 1)
    print(f"Avg nodes/graph: {avg_nodes:.1f}")

    import csv
    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=labels_list[0].keys())
        w.writeheader()
        w.writerows(labels_list)
    print(f"Labels saved to {os.path.join(OUTPUT_DIR, 'labels.csv')}")

    if train_data:
        train_path = os.path.join(OUTPUT_DIR, "training_data.pt")
        torch.save(train_data, train_path)
        print(f"Training data saved: {len(train_data)} graphs -> {train_path}")

    return train_data

if __name__ == "__main__":
    data = main()
    if data and len(data) >= 100:
        print("\nDataset ready! Run train_gnn.py to train the GNN model.")
