"""Build true 12-D graph dataset by synthesizing physical CAD variations on ABC dataset.
Uses rules engine to produce mathematically correct label indicators without skew.
"""
import os, sys, yaml, random, math, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch_geometric.data import Data

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly
from backend.ml.gnn_model import DFMGNN

FEAT_DIR = "F:/Varroc/data/abc_raw/feat"
OUTPUT_DIR = "F:/Varroc/data/processed"
MODEL_DIR = "F:/Varroc/data/models"
MAX_FILE_MB = 2.0
MAX_GRAPHS = 4000

TYPE_MAP = {"Plane": 0, "Cylinder": 1, "Sphere": 2, "Cone": 3, "Torus": 4, "BSpline": 5}
FT_MAP_ = {"Plane": FaceType.PLANE, "Cylinder": FaceType.CYLINDER, "Sphere": FaceType.SPHERE,
           "Cone": FaceType.CONE, "Torus": FaceType.TORUS, "BSpline": FaceType.BSPLINE}

yaml_loader = yaml.CLoader if hasattr(yaml, 'CLoader') else yaml.Loader
random.seed(42)

def estimate_area_from_verts(coefficients, vert_count, bbox_dims):
    if vert_count < 4:
        return 10.0
    x, y, z = bbox_dims
    return max((x * y * z) ** 0.667 * (vert_count / 1000.0), 5.0)

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

def compute_14d_graph_stats(surfaces):
    """
    2 additional graph-level stats, broadcast to every node.
    Both passed correlation gate (|r| >= 0.10) on 491-sample test.
    """
    plane = [s for s in surfaces if s.get('type') == 'Plane']
    
    # Feature 13: min draft angle among Plane faces only, normalized
    plane_drafts = [s['draft_angle'] for s in plane 
                    if 'draft_angle' in s]
    min_plane_draft = min(plane_drafts) / 45.0 if plane_drafts else 1.0
    
    # Feature 14: fraction of all faces that are Plane
    frac_plane = len(plane) / len(surfaces) if surfaces else 0.0
    
    return [min_plane_draft, frac_plane]

def surfaces_to_part_and_graph(surfs, curves, part_id, nominal_wall, process, material):
    n = len(surfs)
    if n < 3:
        return None, None

    # Calculate bounding box from centroids
    locs = [s.get("location", [0.0, 0.0, 0.0]) for s in surfs]
    xs = [l[0] for l in locs]
    ys = [l[1] for l in locs]
    zs = [l[2] for l in locs]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    dx = max(max_x - min_x, 1.0)
    dy = max(max_y - min_y, 1.0)
    dz = max(max_z - min_z, 1.0)
    max_dim = max(dx, dy, dz)
    scale = random.uniform(100.0, 150.0) / max_dim

    bbox_str = f"{dx*scale:.1f} x {dy*scale:.1f} x {dz*scale:.1f}"

    # Determine if this part will have synthetic defects to ensure balanced labels
    is_defective_part = random.random() < 0.35
    defective_faces = set()
    if is_defective_part:
        num_def = max(1, int(n * random.uniform(0.1, 0.3)))
        defective_faces = set(random.sample(range(n), min(num_def, n)))

    faces_meta = []
    vert_to_surfaces = {}

    for i, s in enumerate(surfs):
        st = s.get("type", "Other")
        radius = s.get("radius") or 0.0
        coeffs = s.get("coefficients", [])
        verts = s.get("vert_indices", [])
        location = s.get("location", [0.0, 0.0, 0.0])
        scaled_loc = [location[0] * scale, location[1] * scale, location[2] * scale]

        is_def = i in defective_faces

        # Curvatures
        curv_min, curv_max = estimate_curvature(st, radius, coeffs)

        # Estimate area and scale it
        raw_area = estimate_area_from_verts(coeffs, len(verts), (dx, dy, dz))
        area = raw_area * (scale ** 2)

        # 1. Synthesize thickness (normal wall thickness vs thin/thick defects)
        thickness = nominal_wall
        if is_def and random.random() < 0.5:
            # Thin wall violation (rule min thickness is usually 1.0mm-1.5mm)
            thickness = random.uniform(0.3, 0.8)
        elif is_def:
            # Thick wall violation
            thickness = random.uniform(5.5, 7.5)
        else:
            # Normal fluctuation
            thickness += random.normalvariate(0, 0.1)
            thickness = max(1.2, min(thickness, 4.5))

        # 2. Synthesize Radius
        radius_val = 0.0
        if radius and radius > 0:
            radius_val = radius * scale
            if is_def and random.random() < 0.3:
                # Fillet too small defect
                radius_val = random.uniform(0.1, 0.4)
        else:
            # For non-curved faces, default is 0.0
            radius_val = 0.0

        # 3. Synthesize Draft Angle
        draft = 0.0
        if st == "Plane":
            # Nominal draft is 1.5 to 3.0 deg
            draft = random.uniform(1.5, 3.5)
            if is_def and random.random() < 0.6:
                # Draft angle violation
                draft = random.uniform(0.0, 0.4)
        else:
            draft = 0.0

        # 4. Synthesize Depth/Width (aspect ratios)
        width = math.sqrt(area) if area > 0 else 5.0
        depth = width * random.uniform(0.3, 1.5)
        if st in ("Cylinder", "Cone") and is_def and random.random() < 0.5:
            # High aspect ratio / deep pocket defect
            depth = width * random.uniform(3.5, 6.0)

        # 5. Synthesize Parent Wall Thickness (to evaluate Rib thickness rules)
        pwt = nominal_wall
        # For ribs, thickness should be <= 0.6 * pwt
        is_rib = (st == "Plane" and area < 400.0)
        if is_rib:
            if not is_def:
                thickness = nominal_wall * random.uniform(0.4, 0.55)
            else:
                # Rib too thick defect
                thickness = nominal_wall * random.uniform(0.8, 1.2)

        ft = FT_MAP_.get(st, FaceType.OTHER)
        faces_meta.append(FaceGeometry(
            face_id=f"{part_id}_face_{i:04d}",
            face_type=ft,
            area_mm2=float(area),
            thickness_mm=float(thickness),
            draft_angle_deg=float(draft),
            radius_mm=float(radius_val),
            depth_mm=float(depth),
            width_mm=float(width),
            sw_feature_name=f"Surface_{i}",
            sw_feature_type=st,
            curvature_min=float(curv_min),
            curvature_max=float(curv_max),
            centroid_x=float(scaled_loc[0]),
            centroid_y=float(scaled_loc[1]),
            centroid_z=float(scaled_loc[2]),
            parent_wall_thickness_mm=float(pwt)
        ))

        for v in verts:
            if v not in vert_to_surfaces:
                vert_to_surfaces[v] = []
            vert_to_surfaces[v].append(i)

    # Build shared vertex adjacency
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

    # Part metadata
    part = PartMetadata(
        filename=f"abc_{part_id}",
        bounding_box_mm=bbox_str,
        face_count=n,
        faces=faces_meta,
        process=process,
        material=material,
        nominal_wall_mm=nominal_wall
    )

    return part, edge_index

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Scanning candidates in ABC feat dataset ===")
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
                candidates.append((part_dir, fp))
                break # Only one file per part directory

    print(f"Found {len(candidates)} candidate parts under {MAX_FILE_MB}MB in {time.time()-t0:.1f}s")

    feature_extractor = DFMGNN(12, 128, 1)
    graphs = []
    labels_list = []
    skipped = 0

    print("\n=== Parsing geometry and evaluating rules ===")
    t1 = time.time()
    for idx, (part_id, fp) in enumerate(candidates):
        if len(graphs) >= MAX_GRAPHS:
            break
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t1
            rate = idx / elapsed
            print(f"  [{idx}/{min(len(candidates), MAX_GRAPHS)}] {len(graphs)} graphs built, rate: {rate:.1f} parts/s")

        try:
            with open(fp, 'rb') as f:
                data = yaml.load(f, Loader=yaml_loader)
        except Exception:
            skipped += 1
            continue

        surfs = data.get("surfaces", [])
        curves = data.get("curves", [])
        n = len(surfs)
        if n < 3 or n > 250:
            skipped += 1
            continue

        # Choose a process and nominal wall thickness
        process = random.choice(["injection_moulding", "die_cast_al", "die_cast_zn", "die_cast_mg"])
        material = random.choice(["ABS", "PC", "Nylon"]) if process == "injection_moulding" else random.choice(["Aluminum", "Zinc", "Magnesium"])
        nominal_wall = random.choice([1.5, 2.0, 2.5, 3.0])

        res = surfaces_to_part_and_graph(surfs, curves, part_id, nominal_wall, process, material)
        if res is None:
            skipped += 1
            continue
        part, edge_index = res

        # Run the DFM rules engine
        violations = engine.validate(part)
        has_defect = any(v.severity.value in ("CRITICAL", "WARNING") for v in violations)
        is_def = int(has_defect)

        # Extract true 12-D features using the model's feature extractor
        x_list = []
        for face in part.faces:
            # We use f.model_dump() or dict
            feats = feature_extractor.get_node_features(face.model_dump())
            x_list.append(feats)

        # Build formatted surfaces for 14-D graph stats
        surfaces_formatted = []
        for face in part.faces:
            surfaces_formatted.append({
                "type": face.face_type.value if hasattr(face.face_type, "value") else str(face.face_type),
                "draft_angle": face.draft_angle_deg
            })

        extra = compute_14d_graph_stats(surfaces_formatted)

        for i in range(len(x_list)):
            x_list[i] = x_list[i] + extra

        assert len(x_list[0]) == 14, f"Expected 14-D, got {len(x_list[0])}"

        x = torch.tensor(x_list, dtype=torch.float32)
        y = torch.tensor([float(is_def)], dtype=torch.float32)

        graph = Data(x=x, edge_index=edge_index, y=y, part_id=part_id)
        graph.batch = torch.zeros(n, dtype=torch.long)
        graphs.append(graph)

        labels_list.append({
            "part_id": part_id,
            "face_count": n,
            "violation_count": len(violations),
            "is_defective": is_def,
            "process": process,
            "nominal_wall": nominal_wall
        })

    t2 = time.time()
    defective = sum(1 for g in graphs if g.y.item() == 1)
    clean = len(graphs) - defective
    pos_ratio = (defective / len(graphs)) * 100 if graphs else 0.0
    clean_ratio = (clean / len(graphs)) * 100 if graphs else 0.0
    pos_weight_val = clean / max(defective, 1)

    print(f"\nGenerated {len(graphs)} true 14-D face-graphs in {t2-t1:.0f}s")
    print(f"Built: {len(graphs)} graphs | Defective: {pos_ratio:.1f}% | Clean: {clean_ratio:.1f}% | pos_weight: {pos_weight_val:.4f}")
    print(f"Skipped: {skipped}")

    pt_path = os.path.join(OUTPUT_DIR, "real_cad_training_data_14d.pt")
    torch.save(graphs, pt_path)
    print(f"14-D graph dataset saved: {len(graphs)} graphs -> {pt_path}")

    csv_path = os.path.join(OUTPUT_DIR, "real_cad_labels_14d.csv")
    with open(csv_path, "w", newline="") as f:
        if labels_list:
            w = csv.DictWriter(f, fieldnames=list(labels_list[0].keys()))
            w.writeheader()
            w.writerows(labels_list)
    print(f"Labels saved to {csv_path}")

if __name__ == "__main__":
    main()
