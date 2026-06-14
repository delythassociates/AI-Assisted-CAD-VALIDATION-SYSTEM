"""
build_dataset_full.py
=====================
Single-threaded build of the full 12-D graph dataset from all ABC YMLs.
- No MAX_GRAPHS cap (processes all ~7,168 candidates)
- tqdm progress bar — no multiprocessing (Windows PyTorch stability)
- Skip audit printed every 500 files
- Saves to data/processed/real_cad_training_data_full.pt (non-destructive)
- Prints full Step-6 report at end

Run:
    python scripts/build_dataset_full.py
"""
import os, sys, yaml, random, math, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Max surfaces a single shared vertex can connect — prevents O(n²) adjacency explosion
# on BSpline meshes where hub vertices touch hundreds of surfaces.
MAX_SURFS_PER_VERTEX = 30

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly  # MUST import to register rules via decorators
from backend.ml.gnn_model import DFMGNN

# ── Config ────────────────────────────────────────────────────────────────────
FEAT_DIR   = "F:/Varroc/data/abc_raw/feat"
OUTPUT_DIR = "F:/Varroc/data/processed"
OUTPUT_PT  = os.path.join(OUTPUT_DIR, "real_cad_training_data_full.pt")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "real_cad_labels_full.csv")
MAX_FILE_MB = 2.0
# NOTE: No MAX_GRAPHS cap — process everything.

FT_MAP_ = {
    "Plane":    FaceType.PLANE,
    "Cylinder": FaceType.CYLINDER,
    "Sphere":   FaceType.SPHERE,
    "Cone":     FaceType.CONE,
    "Torus":    FaceType.TORUS,
    "BSpline":  FaceType.BSPLINE,
}

yaml_loader = yaml.CLoader if hasattr(yaml, "CLoader") else yaml.Loader
random.seed(42)

# ── Geometry helpers ──────────────────────────────────────────────────────────
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


def surfaces_to_part_and_graph(surfs, curves, part_id, nominal_wall, process, material):
    n = len(surfs)
    if n < 3:
        return None, None

    locs   = [s.get("location", [0.0, 0.0, 0.0]) for s in surfs]
    xs, ys, zs = [l[0] for l in locs], [l[1] for l in locs], [l[2] for l in locs]
    dx = max(max(xs) - min(xs), 1.0)
    dy = max(max(ys) - min(ys), 1.0)
    dz = max(max(zs) - min(zs), 1.0)
    scale = random.uniform(100.0, 150.0) / max(dx, dy, dz)
    bbox_str = f"{dx*scale:.1f} x {dy*scale:.1f} x {dz*scale:.1f}"

    is_defective_part = random.random() < 0.35
    defective_faces = set()
    if is_defective_part:
        num_def = max(1, int(n * random.uniform(0.1, 0.3)))
        defective_faces = set(random.sample(range(n), min(num_def, n)))

    faces_meta      = []
    vert_to_surfaces = {}

    for i, s in enumerate(surfs):
        st      = s.get("type", "Other")
        radius  = s.get("radius") or 0.0
        coeffs  = s.get("coefficients", [])
        verts   = s.get("vert_indices", [])
        location = s.get("location", [0.0, 0.0, 0.0])
        scaled_loc = [location[0] * scale, location[1] * scale, location[2] * scale]

        is_def = i in defective_faces
        curv_min, curv_max = estimate_curvature(st, radius, coeffs)
        raw_area = estimate_area_from_verts(coeffs, len(verts), (dx, dy, dz))
        area     = raw_area * (scale ** 2)

        # Thickness
        thickness = nominal_wall
        if is_def and random.random() < 0.5:
            thickness = random.uniform(0.3, 0.8)
        elif is_def:
            thickness = random.uniform(5.5, 7.5)
        else:
            thickness += random.normalvariate(0, 0.1)
            thickness = max(1.2, min(thickness, 4.5))

        # Radius
        radius_val = 0.0
        if radius and radius > 0:
            radius_val = radius * scale
            if is_def and random.random() < 0.3:
                radius_val = random.uniform(0.1, 0.4)

        # Draft angle
        draft = 0.0
        if st == "Plane":
            draft = random.uniform(1.5, 3.5)
            if is_def and random.random() < 0.6:
                draft = random.uniform(0.0, 0.4)

        # Depth / Width
        width = math.sqrt(area) if area > 0 else 5.0
        depth = width * random.uniform(0.3, 1.5)
        if st in ("Cylinder", "Cone") and is_def and random.random() < 0.5:
            depth = width * random.uniform(3.5, 6.0)

        # Rib logic
        pwt    = nominal_wall
        is_rib = (st == "Plane" and area < 400.0)
        if is_rib:
            thickness = (nominal_wall * random.uniform(0.4, 0.55)
                         if not is_def else nominal_wall * random.uniform(0.8, 1.2))

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
            vert_to_surfaces.setdefault(v, []).append(i)

    # Adjacency from shared vertices.
    # Skip hub vertices shared by >MAX_SURFS_PER_VERTEX surfaces to avoid
    # O(n^2) combinatorial explosion on complex BSpline meshes.
    adj_set = set()
    for v, s_list in vert_to_surfaces.items():
        u = list(set(s_list))
        if len(u) > MAX_SURFS_PER_VERTEX:
            continue   # pathological hub vertex — skip
        for ii in range(len(u)):
            for jj in range(ii + 1, len(u)):
                a, b = u[ii], u[jj]
                adj_set.add((a, b) if a < b else (b, a))

    edge_list  = [[a, b] for a, b in adj_set] + [[b, a] for a, b in adj_set]
    edge_index = (torch.tensor(edge_list, dtype=torch.long).t()
                  if edge_list else torch.zeros((2, 0), dtype=torch.long))

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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Collect all candidates ──
    print("=== Scanning ABC feat directory ===")
    t_scan = time.time()
    candidates = []
    for part_dir in sorted(os.listdir(FEAT_DIR)):
        dp = os.path.join(FEAT_DIR, part_dir)
        if not os.path.isdir(dp):
            continue
        for fname in os.listdir(dp):
            if not fname.endswith(".yml"):
                continue
            fp = os.path.join(dp, fname)
            if os.path.getsize(fp) < MAX_FILE_MB * 1024 * 1024:
                candidates.append((part_dir, fp))
                break   # one YML per part directory

    print(f"  Found {len(candidates)} candidates in {time.time()-t_scan:.1f}s")

    # ── Check if source_path is tracked in existing dataset ──
    existing_pt = os.path.join(OUTPUT_DIR, "real_cad_training_data.pt")
    if os.path.exists(existing_pt):
        existing = torch.load(existing_pt, map_location="cpu", weights_only=False)
        existing_sources = set(
            g.source_path for g in existing if hasattr(g, "source_path")
        )
        print(f"  Existing dataset: {len(existing)} graphs, "
              f"{len(existing_sources)} with tracked source_path")
        if existing_sources:
            before = len(candidates)
            candidates = [(pid, fp) for pid, fp in candidates
                          if fp not in existing_sources]
            print(f"  Skipping {before - len(candidates)} already-processed YMLs")
        else:
            print("  No source_path on existing graphs — processing all 7,168 "
                  "(duplicates acceptable at this scale)")
        del existing
    print()

    # ── Feature extractor (shared, instantiated once) ──
    feature_extractor = DFMGNN(12, 128, 1)

    graphs      = []
    labels_list = []
    skip_reasons = {
        "too_few_surfaces":  0,
        "too_many_surfaces": 0,
        "parse_error":       0,
        "build_error":       0,
        "feature_error":     0,
        "rules_error":       0,
        "timeout":           0,
    }

    print("=== Building graphs (single-threaded) ===")
    t1 = time.time()

    for i, (part_id, fp) in enumerate(tqdm(candidates, desc="Building graphs", unit="yml")):

        # ── Progress audit every 500 ──
        if i > 0 and i % 500 == 0:
            tqdm.write(
                f"[{i}/{len(candidates)}] graphs={len(graphs)} | skips={skip_reasons}"
            )

        # ── Parse YAML ──
        try:
            with open(fp, "rb") as f:
                data = yaml.load(f, Loader=yaml_loader)
        except Exception:
            skip_reasons["parse_error"] += 1
            continue

        surfs = data.get("surfaces", [])
        n     = len(surfs)
        if n < 3:
            skip_reasons["too_few_surfaces"] += 1
            continue
        if n > 250:
            skip_reasons["too_many_surfaces"] += 1
            continue

        # ── Randomise process / material ──
        process = random.choice(
            ["injection_moulding", "die_cast_al", "die_cast_zn", "die_cast_mg"]
        )
        material = (random.choice(["ABS", "PC", "Nylon"])
                    if process == "injection_moulding"
                    else random.choice(["Aluminum", "Zinc", "Magnesium"]))
        nominal_wall = random.choice([1.5, 2.0, 2.5, 3.0])

        # ── Build part + edge_index ──
        try:
            res = surfaces_to_part_and_graph(
                surfs, data.get("curves", []),
                part_id, nominal_wall, process, material
            )
        except Exception:
            skip_reasons["build_error"] += 1
            continue

        if res is None or res[0] is None:
            skip_reasons["build_error"] += 1
            continue
        part, edge_index = res

        # ── Run rules engine ──
        try:
            violations = engine.validate(part)
        except Exception:
            skip_reasons["rules_error"] += 1
            continue

        has_defect = any(v.severity.value in ("CRITICAL", "WARNING") for v in violations)
        is_def     = int(has_defect)

        # ── Extract 12-D node features ──
        try:
            x_list = [feature_extractor.get_node_features(face.model_dump())
                      for face in part.faces]
            x = torch.tensor(x_list, dtype=torch.float32)
        except Exception:
            skip_reasons["feature_error"] += 1
            continue

        y     = torch.tensor([float(is_def)], dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, y=y, part_id=part_id)
        graph.batch = torch.zeros(n, dtype=torch.long)
        graphs.append(graph)

        labels_list.append({
            "part_id":         part_id,
            "face_count":      n,
            "violation_count": len(violations),
            "is_defective":    is_def,
            "process":         process,
            "nominal_wall":    nominal_wall,
        })

    t2 = time.time()

    # ── Step 4: Class balance ──
    labels     = [g.y.item() for g in graphs]
    pos        = sum(labels)
    neg        = len(labels) - pos
    pos_weight = neg / pos if pos > 0 else 1.0

    # ── Step 6: Report ──
    node_counts = [g.x.shape[0] for g in graphs]
    print("\n" + "=" * 65)
    print("  STEP 6 REPORT")
    print("=" * 65)
    print(f"  Total YMLs attempted    : {len(candidates)}")
    print(f"  Successfully built      : {len(graphs)}")
    print(f"  Total skipped           : {sum(skip_reasons.values())}")
    print(f"  Build time              : {t2-t1:.0f}s  ({(t2-t1)/60:.1f} min)")
    print()
    print("  Skip breakdown:")
    for reason, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        pct = cnt / len(candidates) * 100 if candidates else 0
        print(f"    {reason:22s}: {cnt:5d}  ({pct:.1f}%)")
    print()
    print("  Class distribution:")
    print(f"    Defective (1) : {pos:.0f}  ({100*pos/len(labels):.1f}%)")
    print(f"    Clean     (0) : {neg:.0f}  ({100*neg/len(labels):.1f}%)")
    print(f"    pos_weight (neg/pos) = {pos_weight:.4f}  <-- use in BCEWithLogitsLoss")
    print()
    print("  Node count stats:")
    print(f"    min  = {min(node_counts)}")
    print(f"    max  = {max(node_counts)}")
    print(f"    mean = {sum(node_counts)/len(node_counts):.1f}")
    print("=" * 65)

    if pos / len(labels) > 0.60:
        print()
        print("  NOTE: Defective rate still above 60%.")
        print("  Consider lowering is_defective_part threshold from 0.35 to 0.20")
        print("  in surfaces_to_part_and_graph() before the next build run")
        print("  to get closer to 50/50 balance.")

    # ── Step 5: Save ──
    print(f"\n  Saving to {OUTPUT_PT} ...")
    torch.save(graphs, OUTPUT_PT)
    print(f"  Saved {len(graphs)} graphs.")

    if labels_list:
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(labels_list[0].keys()))
            w.writeheader()
            w.writerows(labels_list)
        print(f"  Labels CSV -> {OUTPUT_CSV}")

    # ── Sanity check ──
    print("\n  Sanity check (first 5 graphs):")
    for g in graphs[:5]:
        assert g.x.shape[1] == 12,           f"Expected 12 features, got {g.x.shape[1]}"
        assert g.y.shape == torch.Size([1]),  f"Bad label shape: {g.y.shape}"
        print(f"    part_id={g.part_id}  nodes={g.x.shape[0]}"
              f"  edges={g.edge_index.shape[1]}  label={g.y.item():.0f}")

    print("\nDone. Bring the Step 6 report to generate the retrain prompt.")


if __name__ == "__main__":
    main()
