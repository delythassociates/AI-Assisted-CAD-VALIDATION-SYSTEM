"""Run DFM rules on ABC dataset parts to generate training labels."""
import os, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly

import pandas as pd
import torch
from torch_geometric.data import Data

STAT_DIR = "F:/Varroc/data/abc_raw/stat"
GRAPH_PATH = "F:/Varroc/data/processed/graphs/abc_graphs.pt"
OUTPUT_LABELS = "F:/Varroc/data/labels/abc_labels.csv"
OUTPUT_TRAIN = "F:/Varroc/data/labels/training_data.pt"

TYPE_MAP_NAME = {
    "Plane": FaceType.PLANE, "Cylinder": FaceType.CYLINDER, "Sphere": FaceType.SPHERE,
    "Cone": FaceType.CONE, "Torus": FaceType.TORUS
}

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

def stat_to_part_metadata(part_id: str, stat_data: dict) -> PartMetadata:
    surfs = stat_data.get("surfs", [])
    bbox = stat_data.get("bbox", [0]*9)
    volume = stat_data.get("volume", 0)
    surface = stat_data.get("surface", 0)

    faces = []
    for i, s_type in enumerate(surfs):
        ft = TYPE_MAP_NAME.get(s_type, FaceType.OTHER)
        faces.append(FaceGeometry(
            face_id=f"{part_id}_face_{i:04d}",
            face_type=ft,
            area_mm2=surface / max(len(surfs), 1),
            thickness_mm=2.0,
            draft_angle_deg=2.0,
            radius_mm=1.0,
            depth_mm=10.0,
            width_mm=10.0,
            sw_feature_name=f"Surface_{i}",
            sw_feature_type=s_type
        ))

    bounding_box = f"{abs(bbox[3]-bbox[0]):.1f} x {abs(bbox[4]-bbox[1]):.1f} x {abs(bbox[5]-bbox[2]):.1f}"

    return PartMetadata(
        filename=f"abc_{part_id}",
        bounding_box_mm=bounding_box,
        face_count=len(faces),
        faces=faces
    )

def load_graphs():
    if os.path.exists(GRAPH_PATH):
        return torch.load(GRAPH_PATH, weights_only=False)
    return []

def main():
    os.makedirs(os.path.dirname(OUTPUT_LABELS), exist_ok=True)
    parts = get_all_stat_files()
    print(f"Loaded {len(parts)} stat files")

    graphs = load_graphs()
    print(f"Loaded {len(graphs)} graphs")

    labels = []
    training_data = []

    for i, (part_id, stat_data) in enumerate(parts):
        if i % 500 == 0:
            print(f"Processing {i}/{len(parts)}...")

        part = stat_to_part_metadata(part_id, stat_data)
        violations = engine.validate(part)

        is_defective = len(violations) > 0
        severity_map = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
        max_sev = max((severity_map.get(v.severity.value, 0) for v in violations), default=0)

        labels.append({
            "part_id": part_id,
            "face_count": part.face_count,
            "violation_count": len(violations),
            "has_critical": any(v.severity.value == "CRITICAL" for v in violations),
            "has_warning": any(v.severity.value == "WARNING" for v in violations),
            "is_defective": int(is_defective),
            "severity_score": max_sev,
            "violation_ids": ",".join(v.rule_id for v in violations),
            "volume": stat_data.get("volume", 0),
            "surface_area": stat_data.get("surface", 0),
            "face_types": ",".join(stat_data.get("surfs", [])),
            "gaussian_curvature": stat_data.get("gaus_curv", 0),
            "mean_curvature": stat_data.get("mean_curv", 0)
        })

        if i < len(graphs) and graphs[i] is not None:
            graph = graphs[i]
            graph.y = torch.tensor([int(is_defective)], dtype=torch.float32)
            training_data.append(graph)

    df = pd.DataFrame(labels)
    df.to_csv(OUTPUT_LABELS, index=False)
    print(f"Labels saved: {len(labels)} parts to {OUTPUT_LABELS}")
    print(f"  Defective: {df['is_defective'].sum()}/{len(df)} ({df['is_defective'].mean()*100:.1f}%)")
    print(f"  Critical: {df['has_critical'].sum()}/{len(df)}")

    torch.save(training_data, OUTPUT_TRAIN)
    print(f"Training data saved: {len(training_data)} graphs to {OUTPUT_TRAIN}")

if __name__ == "__main__":
    main()
