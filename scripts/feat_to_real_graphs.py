"""Convert ABC Dataset FEAT files to real face-adjacency PyG graphs.
Each surface -> node, shared vertex indices -> edges."""
import os, yaml, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch_geometric.data import Data

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly

FEAT_DIR = "F:/Varroc/data/abc_raw/feat"
OUTPUT_DIR = "F:/Varroc/data/processed"
MODEL_DIR = "F:/Varroc/data/models"

TYPE_MAP = {"Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3, "Torus": 4, "BSpline": 5}
FT_MAP_ = {"Plane": FaceType.PLANE, "Cylinder": FaceType.CYLINDER, "Sphere": FaceType.SPHERE,
           "Cone": FaceType.CONE, "Torus": FaceType.TORUS, "BSpline": FaceType.BSPLINE}

yaml_loader = yaml.CLoader if hasattr(yaml, 'CLoader') else yaml.Loader

def estimate_area_from_verts(coefficients, vert_count, bbox_dims):
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
    if face_type == "Cone":
        return 0.0, 0.0
    return 0.5, 0.5

def surfaces_to_graph(surfaces, curves, part_id):
    n = len(surfaces)
    if n < 3:
        return None, None

    x_list = []
    faces_meta = []
    vert_to_surfaces = {}

    bbox_dims = (100.0, 100.0, 100.0)

    for i, s in enumerate(surfaces):
        st = s.get("type", "Other")
        radius = s.get("radius") or 0.0
        coeffs = s.get("coefficients", [])
        verts = s.get("vert_indices", [])
        location = s.get("location", [0, 0, 0])

        ft_idx = TYPE_MAP.get(st, 6)
        vert_count = len(verts)
        area = estimate_area_from_verts(coeffs, vert_count, bbox_dims)
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

        ft = FT_MAP_.get(st, FaceType.OTHER)
        faces_meta.append(FaceGeometry(
            face_id="%s_face_%04d" % (part_id, i),
            face_type=ft, area_mm2=float(area), thickness_mm=thickness,
            draft_angle_deg=draft, radius_mm=float(radius_val),
            depth_mm=depth, width_mm=width,
            sw_feature_name="Surface_%d" % i, sw_feature_type=st,
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
        for i in range(len(unique_s)):
            for j in range(i + 1, len(unique_s)):
                a, b = unique_s[i], unique_s[j]
                adj_set.add((a, b) if a < b else (b, a))

    edge_list = []
    for a, b in adj_set:
        edge_list.append([a, b])
        edge_list.append([b, a])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, part_id=part_id, num_surfaces=n)
    return graph, faces_meta

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    t0 = time.time()
    feat_parts = []
    for part_dir in sorted(os.listdir(FEAT_DIR)):
        dp = os.path.join(FEAT_DIR, part_dir)
        if os.path.isdir(dp):
            for fname in os.listdir(dp):
                if fname.endswith('.yml'):
                    fpath = os.path.join(dp, fname)
                    with open(fpath, 'r') as f:
                        data = yaml.load(f, Loader=yaml_loader)
                    feat_parts.append((part_dir, data))
                    break
    t1 = time.time()
    print("Loaded %d feat files in %.1fs" % (len(feat_parts), t1 - t0))

    graphs = []
    labels_list = []
    train_data = []
    total_surfs = 0
    total_edges = 0

    for idx, (part_id, data) in enumerate(feat_parts):
        if idx % 500 == 0:
            print("Processing %d/%d... (%.0fs)" % (idx, len(feat_parts), time.time() - t1))

        surfs = data.get("surfaces", [])
        curves = data.get("curves", [])
        graph, faces_meta = surfaces_to_graph(surfs, curves, part_id)
        if graph is None:
            continue

        bb = [0, 0, 0, 100, 100, 100]
        part = PartMetadata(
            filename="abc_%s" % part_id,
            bounding_box_mm="100 x 100 x 100",
            face_count=len(faces_meta),
            faces=faces_meta
        )

        violations = engine.validate(part)
        has_real = any(v.severity.value in ("CRITICAL", "WARNING") for v in violations)
        is_def = int(has_real)

        graph.y = torch.tensor([float(is_def)], dtype=torch.float32)
        train_data.append(graph)
        graphs.append(graph)
        total_surfs += len(surfs)
        total_edges += graph.edge_index.shape[1] // 2

        labels_list.append({
            "part_id": part_id,
            "face_count": len(surfs),
            "violation_count": len(violations),
            "is_defective": is_def,
        })

    t2 = time.time()
    defective = sum(1 for g in train_data if g.y.item() == 1)
    print("\nGenerated %d real face-graphs in %.0fs" % (len(graphs), t2 - t1))
    print("Defective: %d/%d (%.1f%%)" % (defective, len(train_data), defective/max(len(train_data),1)*100))
    print("Avg surfaces/graph: %.1f" % (total_surfs / max(len(graphs), 1)))
    print("Avg edges/graph: %.1f" % (total_edges / max(len(graphs), 1)))

    import csv
    csv_path = os.path.join(OUTPUT_DIR, "real_labels.csv")
    with open(csv_path, "w", newline="") as f:
        if labels_list:
            w = csv.DictWriter(f, fieldnames=list(labels_list[0].keys()))
            w.writeheader()
            w.writerows(labels_list)
    print("Labels saved to", csv_path)

    pt_path = os.path.join(OUTPUT_DIR, "real_training_data.pt")
    torch.save(train_data, pt_path)
    print("Training data saved: %d graphs -> %s" % (len(train_data), pt_path))
    return train_data

if __name__ == "__main__":
    import math
    data = main()
