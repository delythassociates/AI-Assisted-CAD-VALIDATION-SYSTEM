"""
injection.py — Injection Moulding DFM Rules

Existing rules: INJ-WALL-001..003, INJ-DRAFT-001..002, INJ-CORNER-001,
                INJ-HOLE-001, INJ-FLOW-001, INJ-RIB-001..003, INJ-BOSS-001..003
New rules:      INJ-UNDERCUT-001, INJ-SINK-001, INJ-PARTING-001
"""

import math
from .engine import engine
from ..core.models import PartMetadata, Violation, Severity, Category


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pull_direction(raw: str) -> tuple:
    """Parse pull direction string to a unit vector.
    Accepts: '+Z', '-Z', '+Y', '-Y', '+X', '-X', 'auto' (defaults to +Z),
             or '(x,y,z)' tuple format.
    """
    s = (raw or "auto").strip().upper()

    axis_map = {
        "+Z": (0.0, 0.0, 1.0),  "-Z": (0.0, 0.0, -1.0),
        "+Y": (0.0, 1.0, 0.0),  "-Y": (0.0, -1.0, 0.0),
        "+X": (1.0, 0.0, 0.0),  "-X": (-1.0, 0.0, 0.0),
        "Z":  (0.0, 0.0, 1.0),  "Y":  (0.0, 1.0, 0.0),
        "X":  (1.0, 0.0, 0.0),
    }
    if s in axis_map:
        return axis_map[s]

    # Try parsing (x,y,z) format
    try:
        cleaned = s.strip("()[] ")
        parts = [float(p) for p in cleaned.split(",")]
        if len(parts) == 3:
            mag = math.sqrt(sum(p * p for p in parts))
            if mag > 1e-9:
                return tuple(p / mag for p in parts)
    except (ValueError, TypeError):
        pass

    # Default: +Z (mould opens along Z)
    return (0.0, 0.0, 1.0)


def _dot(a: tuple, b: tuple) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Rule 1 — Undercut Detection (Injection)
# ---------------------------------------------------------------------------

@engine.register
def inj_undercut_001(part: PartMetadata) -> list[Violation]:
    """Flag faces whose normals oppose the pull direction by more than 5deg.
    These faces create undercuts requiring side-actions or lifters."""
    violations = []
    pull = _parse_pull_direction(part.pull_direction)
    is_pull_unknown = not part.pull_direction or part.pull_direction.lower() in ("auto", "none", "")

    for face in part.faces:
        normal = (face.normal_x, face.normal_y, face.normal_z)
        dot = _dot(normal, pull)

        # cos(95deg) ~ -0.087 -- face opposes pull by more than 5deg
        if dot < -0.087:
            # Skip cylinders (typically pins/cores, handled separately)
            ftype = getattr(face.face_type, "value", str(face.face_type))
            if ftype in ("Plane", "BSpline", "Cone", "Torus", "Other"):
                angle_deg = round(math.degrees(math.acos(_clamp(dot, -1.0, 1.0))), 1)
                violations.append(Violation(
                    rule_id="INJ-UNDERCUT-001",
                    category=Category.INJECTION_MOLDING.value,
                    severity=Severity.CRITICAL,
                    face_ids=[face.face_id],
                    solidworks_feature_name=face.sw_feature_name,
                    measured_value=f"{angle_deg}deg from pull",
                    required_value="<=90deg (no undercut)",
                    standard_reference="Varroc DFM-INJ-2025 / General mould design",
                    description=(
                        f"Face {face.face_id} is undercut -- normal opposes pull direction "
                        f"by {angle_deg}deg. Requires side-action or redesign."
                    ),
                    fix_suggestion=(
                        "Redesign face to align within 90deg of pull direction, "
                        "or add a side-action/lifter to the mould tool."
                    ),
                    solidworks_fix_path="Modify face geometry or Insert > Features > Draft",
                    unaddressed_risk_score=9,
                    unaddressed_risk_reasoning=(
                        "Undercuts require side-action sliders or lifters, adding 30-50% "
                        "to tooling cost and increasing cycle time."
                    ),
                    status="PENDING" if is_pull_unknown else "ACTIVE",
                ))
    return violations


# ---------------------------------------------------------------------------
# Rule 2 — Sink Mark Risk
# ---------------------------------------------------------------------------

@engine.register
def inj_sink_001(part: PartMetadata) -> list[Violation]:
    """Flag faces where local wall thickness exceeds 1.5x nominal and >3mm.
    Thick sections cool slower, creating visible surface depressions (sink marks)."""
    violations = []
    nominal = part.nominal_wall_mm or 1.5

    for face in part.faces:
        if face.thickness_mm is not None and face.thickness_mm > 0 and nominal > 0:
            ratio = face.thickness_mm / nominal
            if ratio > 1.5 and face.thickness_mm > 3.0:
                max_allowed = round(nominal * 1.5, 1)
                violations.append(Violation(
                    rule_id="INJ-SINK-001",
                    category=Category.INJECTION_MOLDING.value,
                    severity=Severity.WARNING,
                    face_ids=[face.face_id],
                    solidworks_feature_name=face.sw_feature_name,
                    measured_value=f"{face.thickness_mm:.2f}mm ({ratio:.1f}x nominal)",
                    required_value=f"<={max_allowed}mm (1.5x nominal {nominal}mm)",
                    standard_reference="Varroc DFM-INJ-2025 / Sink mark prevention",
                    description=(
                        f"Face {face.face_id} wall is {ratio:.1f}x nominal -- "
                        f"sink mark risk on cosmetic surface."
                    ),
                    fix_suggestion=(
                        f"Reduce local wall thickness to <={max_allowed}mm, "
                        f"or core out the section. Add texture to cosmetic face "
                        f"to hide sink if redesign is not feasible."
                    ),
                    solidworks_fix_path="Shell or core-out features to reduce section thickness",
                    unaddressed_risk_score=6,
                    unaddressed_risk_reasoning=(
                        "Thick sections cool slower than surrounding walls, creating "
                        "surface depressions (sink marks) visible on cosmetic surfaces."
                    ),
                ))
    return violations


# ---------------------------------------------------------------------------
# Rule 3 — Parting Line Draft Conflict
# ---------------------------------------------------------------------------

@engine.register
def inj_parting_001(part: PartMetadata) -> list[Violation]:
    """Flag Class A faces at the parting line (normal nearly perpendicular to pull).
    The parting line witness mark will be visible on cosmetic surfaces."""
    violations = []
    pull = _parse_pull_direction(part.pull_direction)
    is_pull_unknown = not part.pull_direction or part.pull_direction.lower() in ("auto", "none", "")
    class_a_set = set(part.class_a_face_ids or [])

    for face in part.faces:
        normal = (face.normal_x, face.normal_y, face.normal_z)
        dot = abs(_dot(normal, pull))

        # Face normal nearly perpendicular to pull -- within 5deg of parting plane
        # cos(85deg) ~ 0.087
        is_parting_candidate = dot < 0.087

        if is_parting_candidate and face.face_id in class_a_set:
            angle_deg = round(math.degrees(math.acos(_clamp(dot, 0.0, 1.0))), 1)
            violations.append(Violation(
                rule_id="INJ-PARTING-001",
                category=Category.INJECTION_MOLDING.value,
                severity=Severity.CRITICAL,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"{angle_deg}deg from parting plane",
                required_value="Class A face must not sit at parting line",
                standard_reference="Varroc DFM-INJ-2025 / Parting line aesthetics",
                description=(
                    f"Face {face.face_id} is a Class A surface at the parting line -- "
                    f"parting line witness mark will be visible."
                ),
                fix_suggestion=(
                    "Move the parting line away from this Class A surface, or add a "
                    "shut-off angle of >=3deg to push the witness mark to a hidden face."
                ),
                solidworks_fix_path="Redesign parting line location in mould tooling design",
                unaddressed_risk_score=8,
                unaddressed_risk_reasoning=(
                    "Parting line witness marks on Class A (cosmetic) surfaces are "
                    "visible to end users and typically cause rejection at quality audit."
                ),
                status="PENDING" if is_pull_unknown else "ACTIVE",
            ))
    return violations


# ---------------------------------------------------------------------------
# DC-UNDERCUT-001 — Undercut Detection (Die Casting)
# Registered here so the die_casting module doesn't duplicate the helper.
# ---------------------------------------------------------------------------

@engine.register
def dc_undercut_001(part: PartMetadata) -> list[Violation]:
    """Undercut detection for die casting -- same geometry logic as injection
    but with die-casting severity and messaging."""
    violations = []
    pull = _parse_pull_direction(part.pull_direction)
    is_pull_unknown = not part.pull_direction or part.pull_direction.lower() in ("auto", "none", "")

    for face in part.faces:
        normal = (face.normal_x, face.normal_y, face.normal_z)
        dot = _dot(normal, pull)

        if dot < -0.087:
            ftype = getattr(face.face_type, "value", str(face.face_type))
            if ftype in ("Plane", "BSpline", "Cone", "Torus", "Other"):
                angle_deg = round(math.degrees(math.acos(_clamp(dot, -1.0, 1.0))), 1)
                violations.append(Violation(
                    rule_id="DC-UNDERCUT-001",
                    category=Category.DIE_CASTING.value,
                    severity=Severity.CRITICAL,
                    face_ids=[face.face_id],
                    solidworks_feature_name=face.sw_feature_name,
                    measured_value=f"{angle_deg}deg from pull",
                    required_value="<=90deg (no undercut)",
                    standard_reference="NADCA Guidelines / Varroc DFM-DIE-2025",
                    description=(
                        f"Face {face.face_id} is undercut -- normal opposes pull direction "
                        f"by {angle_deg}deg. Requires slide-core or redesign."
                    ),
                    fix_suggestion=(
                        "Redesign face to align within 90deg of pull direction, "
                        "or add a slide-core mechanism to the die tool."
                    ),
                    solidworks_fix_path="Modify face geometry or Insert > Features > Draft",
                    unaddressed_risk_score=9,
                    unaddressed_risk_reasoning=(
                        "Undercuts in die casting require slide cores or collapsible cores, "
                        "adding 30-50% to die cost and reducing cycle reliability."
                    ),
                    status="PENDING" if is_pull_unknown else "ACTIVE",
                ))
    return violations
