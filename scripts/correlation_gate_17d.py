"""
correlation_gate_17d.py — Phase 1: Correlation gate for 5 conditional features.

Runs on 500 randomly sampled YMLs from data/abc_raw/feat/.
Uses the EXACT same synthesis pipeline as build_dataset_12d.py (draft/thickness
are synthetic — they are NOT present in raw ABC YML files).
Does NOT build full PyG graphs — fast feature extraction only.

Stop condition:  any feature |r| < 0.10 → print FAIL report, halt, do NOT proceed.
Proceed:         all features |r| >= 0.10 → auto-proceed to Phase 2.
"""

import os
import sys
import glob
import math
import random
import time
import yaml

import numpy as np
from scipy.stats import pointbiserialr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.core.models import PartMetadata, FaceGeometry, FaceType
from backend.rules.engine import engine
from backend.rules import injection, die_casting, cnc, gdt, assembly

# ── Config ────────────────────────────────────────────────────────────────────
FEAT_DIR   = "F:/Varroc/data/abc_raw/feat"
SAMPLE_N   = 500
GATE_MIN_R = 0.10          # minimum |r| to pass
MAX_FILE_MB = 2.0
random.seed(0)
yaml_loader = yaml.CLoader if hasattr(yaml, "CLoader") else yaml.Loader

FT_MAP = {
    "Plane":    FaceType.PLANE,
    "Cylinder": FaceType.CYLINDER,
    "Sphere":   FaceType.SPHERE,
    "Cone":     FaceType.CONE,
    "Torus":    FaceType.TORUS,
    "BSpline":  FaceType.BSPLINE,
}

# ── Synthesis helpers (mirrored from build_dataset_12d.py) ────────────────────

def _estimate_area(coeffs, vert_count, bbox_dims):
    if vert_count < 4:
        return 10.0
    x, y, z = bbox_dims
    return max((x * y * z) ** 0.667 * (vert_count / 1000.0), 5.0)


def _synthesize_faces(surfs, nominal_wall, is_defective_part, defective_faces, scale, bbox_dims):
    """
    Returns a list of dicts with synthesized physical properties.
    Matches build_dataset_12d.py exactly so labels are consistent.
    """
    dx, dy, dz = bbox_dims
    faces = []

    for i, s in enumerate(surfs):
        st = s.get("type", "Other")
        radius = s.get("radius") or 0.0
        coeffs = s.get("coefficients", [])
        verts  = s.get("vert_indices", [])
        location = s.get("location", [0.0, 0.0, 0.0])

        is_def = i in defective_faces

        raw_area = _estimate_area(coeffs, len(verts), (dx, dy, dz))
        area = raw_area * (scale ** 2)

        # ── Thickness ────────────────────────────────────────────────────────
        thickness = nominal_wall
        is_rib = (st == "Plane" and area < 400.0)
        if is_rib:
            thickness = (nominal_wall * random.uniform(0.4, 0.55)
                         if not is_def
                         else nominal_wall * random.uniform(0.8, 1.2))
        elif is_def and random.random() < 0.5:
            thickness = random.uniform(0.3, 0.8)   # thin wall violation
        elif is_def:
            thickness = random.uniform(5.5, 7.5)   # thick wall violation
        else:
            thickness += random.normalvariate(0, 0.1)
            thickness = max(1.2, min(thickness, 4.5))

        # ── Radius ───────────────────────────────────────────────────────────
        if radius and radius > 0:
            radius_val = radius * scale
            if is_def and random.random() < 0.3:
                radius_val = random.uniform(0.1, 0.4)
        else:
            radius_val = 0.0

        # ── Draft angle (Plane faces only) ───────────────────────────────────
        if st == "Plane":
            if is_def and random.random() < 0.6:
                draft = random.uniform(0.0, 0.4)   # draft violation
            else:
                draft = random.uniform(1.5, 3.5)
        else:
            draft = 0.0

        faces.append({
            "type":      st,
            "thickness": thickness,
            "radius":    radius_val,
            "draft":     draft,
            "area":      area,
        })

    return faces


def _build_part_metadata(surfs, faces_synth, part_id, nominal_wall, process, material):
    """Builds a PartMetadata pydantic object for the rules engine."""
    fg_list = []
    for i, (s, f) in enumerate(zip(surfs, faces_synth)):
        st = s.get("type", "Other")
        fg_list.append(FaceGeometry(
            face_id=f"{part_id}_face_{i:04d}",
            face_type=FT_MAP.get(st, FaceType.OTHER),
            area_mm2=float(f["area"]),
            thickness_mm=float(f["thickness"]),
            draft_angle_deg=float(f["draft"]),
            radius_mm=float(f["radius"]),
            depth_mm=float(math.sqrt(f["area"])) * 0.8,
            width_mm=float(math.sqrt(f["area"])),
            sw_feature_name=f"Surface_{i}",
            sw_feature_type=st,
            curvature_min=float(1.0 / f["radius"]) if f["radius"] > 0 else 0.0,
            curvature_max=float(1.0 / (f["radius"] * 2)) if f["radius"] > 0 else 0.0,
            centroid_x=float(s.get("location", [0, 0, 0])[0]),
            centroid_y=float(s.get("location", [0, 0, 0])[1]),
            centroid_z=float(s.get("location", [0, 0, 0])[2]),
            parent_wall_thickness_mm=float(nominal_wall),
        ))
    return PartMetadata(
        filename=f"abc_{part_id}",
        face_count=len(fg_list),
        faces=fg_list,
        process=process,
        material=material,
        nominal_wall_mm=nominal_wall,
    )


# ── 5 conditional stats (Phase 2 candidate features) ─────────────────────────

def compute_conditional_stats(synth_faces):
    """
    Graph-level stats conditioned on face type.
    All values normalised to [0, 1].

    The rules engine checks:
      - draft_angle < 0.5 deg  → Plane faces only
      - thickness  < 0.8 mm   → Plane faces only
      - radius     < 0.3 mm   → Cylinder/Cone faces only
    """
    plane = [f for f in synth_faces if f["type"] == "Plane"]
    cyl   = [f for f in synth_faces if f["type"] in ("Cylinder", "Cone")]

    # dim 13 — min draft angle among Plane faces only (normalised /45)
    plane_drafts = [f["draft"] for f in plane]
    min_plane_draft = min(plane_drafts) / 45.0 if plane_drafts else 1.0

    # dim 14 — fraction of Plane faces violating draft < 0.5 deg
    frac_plane_draft_viol = (
        sum(1 for d in plane_drafts if d < 0.5) / len(plane_drafts)
        if plane_drafts else 0.0
    )

    # dim 15 — min thickness among Plane faces only (normalised /5)
    plane_thick = [f["thickness"] for f in plane if f["thickness"] > 0]
    min_plane_thick = min(plane_thick) / 5.0 if plane_thick else 1.0

    # dim 16 — min radius among Cylinder/Cone faces only (normalised /10)
    cyl_radii = [f["radius"] for f in cyl if f["radius"] > 0]
    min_cyl_radius = min(cyl_radii) / 10.0 if cyl_radii else 1.0

    # dim 17 — fraction of all faces that are Plane
    frac_plane = len(plane) / len(synth_faces) if synth_faces else 0.0

    return [min_plane_draft, frac_plane_draft_viol, min_plane_thick,
            min_cyl_radius, frac_plane]


# ── Main correlation gate ─────────────────────────────────────────────────────

def run_gate():
    print("=" * 65)
    print("   CORRELATION GATE -- 17-D Feature Expansion (Phase 1)")
    print("=" * 65)
    print(f"Scanning {FEAT_DIR} for candidate YMLs ...")
    t0 = time.time()

    # Collect one YML per part dir (same filter as build_dataset_12d.py)
    all_paths = []
    for part_dir in os.listdir(FEAT_DIR):
        dp = os.path.join(FEAT_DIR, part_dir)
        if not os.path.isdir(dp):
            continue
        for fname in os.listdir(dp):
            if not fname.endswith(".yml"):
                continue
            fp = os.path.join(dp, fname)
            if os.path.getsize(fp) < MAX_FILE_MB * 1024 * 1024:
                all_paths.append(fp)
                break

    random.shuffle(all_paths)
    sample = all_paths[:SAMPLE_N]
    print(f"Total candidates: {len(all_paths):,}  |  Sample: {len(sample)}")
    print()

    rows, labels = [], []
    skipped = 0

    for idx, fp in enumerate(sample):
        if idx % 100 == 0 and idx > 0:
            print(f"  [{idx}/{len(sample)}] processed={len(rows)}, "
                  f"skipped={skipped}, "
                  f"elapsed={time.time()-t0:.1f}s")
        try:
            with open(fp, "rb") as f:
                data = yaml.load(f, Loader=yaml_loader)
        except Exception:
            skipped += 1
            continue

        surfs = data.get("surfaces", [])
        n = len(surfs)
        if n < 3 or n > 250:
            skipped += 1
            continue

        # ── Synthesis parameters (same random choices as build_dataset_12d) ──
        process      = random.choice(["injection_moulding", "die_cast_al",
                                       "die_cast_zn", "die_cast_mg"])
        material     = (random.choice(["ABS", "PC", "Nylon"])
                        if process == "injection_moulding"
                        else random.choice(["Aluminum", "Zinc", "Magnesium"]))
        nominal_wall = random.choice([1.5, 2.0, 2.5, 3.0])

        # ── Bounding box + scale ─────────────────────────────────────────────
        locs  = [s.get("location", [0, 0, 0]) for s in surfs]
        xs, ys, zs = [l[0] for l in locs], [l[1] for l in locs], [l[2] for l in locs]
        dx = max(max(xs) - min(xs), 1.0)
        dy = max(max(ys) - min(ys), 1.0)
        dz = max(max(zs) - min(zs), 1.0)
        max_dim = max(dx, dy, dz)
        scale   = random.uniform(100.0, 150.0) / max_dim

        # ── Defective face assignment ─────────────────────────────────────────
        is_defective_part = random.random() < 0.35
        defective_faces   = set()
        if is_defective_part:
            num_def = max(1, int(n * random.uniform(0.1, 0.3)))
            defective_faces = set(random.sample(range(n), min(num_def, n)))

        # ── Synthesize physical face properties ──────────────────────────────
        synth = _synthesize_faces(surfs, nominal_wall, is_defective_part,
                                   defective_faces, scale, (dx, dy, dz))

        # ── Run DFM rules engine for label ────────────────────────────────────
        try:
            part = _build_part_metadata(surfs, synth, fp, nominal_wall,
                                         process, material)
            violations = engine.validate(part)
            has_defect = any(v.severity.value in ("CRITICAL", "WARNING")
                             for v in violations)
        except Exception as e:
            skipped += 1
            continue

        # ── Extract 5 conditional stats ───────────────────────────────────────
        stats = compute_conditional_stats(synth)
        rows.append(stats)
        labels.append(float(has_defect))

    print(f"\nParsed {len(rows)} parts in {time.time()-t0:.1f}s  "
          f"(skipped {skipped})\n")

    if len(rows) < 50:
        print("FATAL: Too few valid samples to run correlation gate. Aborting.")
        sys.exit(1)

    X = np.array(rows,   dtype=float)
    y = np.array(labels, dtype=float)

    feature_names = [
        "min_plane_draft",
        "frac_plane_draft_violation",
        "min_plane_thickness",
        "min_cyl_radius",
        "frac_plane",
    ]

    print(f"=== CORRELATION GATE ({len(y)} samples) ===")
    print(f"Label distribution: {y.mean()*100:.1f}% defective\n")
    print(f"{'Feature':<35}  {'r':>7}  {'p':>8}  {'sat%':>6}  {'gate'}")
    print("-" * 70)

    gate_passed  = True
    weak_features = []

    for i, name in enumerate(feature_names):
        col = X[:, i]
        r, p   = pointbiserialr(y, col)
        sat    = (col >= 1.0).mean() * 100    # saturated-at-max fraction
        flag   = "PASS" if abs(r) >= GATE_MIN_R else "WEAK"
        print(f"{name:<35}  r={r:+.3f}  p={p:.4f}  sat={sat:5.1f}%  {flag}")
        if abs(r) < GATE_MIN_R:
            gate_passed = False
            weak_features.append((name, r, p))

    print("-" * 70)
    print()

    if gate_passed:
        print("GATE: PASSED -- all features |r| >= 0.10")
        print("      Auto-proceeding to Phase 2 (full rebuild + retrain) ...\n")
        return True
    else:
        print("GATE: FAILED -- the following features are too weak to add signal:\n")
        for name, r, p in weak_features:
            print(f"  [X]  {name:35s}  r={r:+.3f}  p={p:.4f}")
        print()
        print("RECOMMENDATION: These features are likely dominated by constant values")
        print("  from the ABC raw geometry (no real draft/thickness fields in YML).")
        print("  The synthetic distribution does not show sufficient label separation.")
        print("  Do NOT proceed to Phase 2. Consider alternative features.")
        return False


if __name__ == "__main__":
    passed = run_gate()
    sys.exit(0 if passed else 1)
