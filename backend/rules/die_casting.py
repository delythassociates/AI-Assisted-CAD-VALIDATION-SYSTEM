import math
from .engine import engine
from ..core.models import PartMetadata, Violation, Severity, Category
from .injection import _parse_pull_direction

def _get_process_limits(process: str) -> tuple[float, float, float, float, float, float]:
    # Returns (min_wall, max_wall, ext_draft, int_draft, min_corner, max_wall_porosity)
    p = (process or "").lower()
    if "zn" in p or "zinc" in p:
        return 0.5, 4.0, 0.5, 1.0, 0.5, 99.0
    if "mg" in p or "magnesium" in p:
        return 0.6, 4.0, 1.0, 1.5, 0.8, 99.0
    # Default to Aluminium (Al)
    return 0.8, 5.0, 1.0, 2.0, 1.0, 6.0

@engine.register
def dc_wall_001(part: PartMetadata) -> list[Violation]:
    violations = []
    min_wall, _, _, _, _, _ = _get_process_limits(part.process)
    for face in part.faces:
        if face.thickness_mm is not None and face.thickness_mm < min_wall:
            violations.append(Violation(
                rule_id="DC-WALL-001",
                category=Category.DIE_CASTING.value,
                severity=Severity.CRITICAL,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"{face.thickness_mm:.2f}mm",
                required_value=f"≥{min_wall:.1f}mm for {part.process}",
                standard_reference="NADCA Product Specifications / Varroc DFM-DIE-2025",
                description=f"Wall thickness {face.thickness_mm:.2f}mm below {min_wall:.1f}mm minimum — cold shut and fill failure risk",
                fix_suggestion=f"Increase wall thickness on face to {min_wall:.1f}mm minimum",
                solidworks_fix_path="Edit Extrusion sketch thickness or Shell feature",
                unaddressed_risk_score=9,
                unaddressed_risk_reasoning="Thin die-cast walls solidify before filling the cavity, causing misruns and cold shuts"
            ))
    return violations

@engine.register
def dc_wall_002(part: PartMetadata) -> list[Violation]:
    violations = []
    _, max_wall, _, _, _, _ = _get_process_limits(part.process)
    for face in part.faces:
        if face.thickness_mm is not None and face.thickness_mm > max_wall:
            violations.append(Violation(
                rule_id="DC-WALL-002",
                category=Category.DIE_CASTING.value,
                severity=Severity.WARNING,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"{face.thickness_mm:.2f}mm",
                required_value=f"≤{max_wall:.1f}mm for {part.process}",
                standard_reference="NADCA Guidelines / Varroc DFM-DIE-2025",
                description=f"Wall thickness {face.thickness_mm:.2f}mm exceeds {max_wall:.1f}mm maximum — porosity and long cycle risk",
                fix_suggestion="Use ribs or core-out to reduce wall thickness to standard limits",
                solidworks_fix_path="Use Shell or create core-out pockets",
                unaddressed_risk_score=7,
                unaddressed_risk_reasoning="Thick sections solidify slowly, creating large shrinkage voids and gaseous porosity"
            ))
    return violations

@engine.register
def dc_wall_003(part: PartMetadata) -> list[Violation]:
    violations = []
    thicknesses = [f.thickness_mm for f in part.faces if f.thickness_mm is not None and f.thickness_mm > 0]
    if len(thicknesses) >= 2:
        mx = max(thicknesses)
        mn = min(thicknesses)
        ratio = mx / mn if mn > 0 else 1.0
        if ratio > 3.0:
            violations.append(Violation(
                rule_id="DC-WALL-003",
                category=Category.DIE_CASTING.value,
                severity=Severity.WARNING,
                face_ids=[f.face_id for f in part.faces if f.thickness_mm is not None],
                measured_value=f"Transition ratio {ratio:.2f}",
                required_value="≤3.0 section change ratio",
                standard_reference="NADCA Guidelines",
                description=f"Section change ratio {ratio:.2f} exceeds 3.0 (Max {mx:.2f}mm / Min {mn:.2f}mm) — shrinkage porosity risk",
                fix_suggestion="Redesign part to have more uniform thickness or use gradual chamfers/tapers",
                solidworks_fix_path="Add draft or draft transitions between thick and thin features",
                unaddressed_risk_score=7,
                unaddressed_risk_reasoning="Sudden wall thickness changes disrupt metal flow and cause cooling shrink defects"
            ))
    return violations

@engine.register
def dc_draft_001(part: PartMetadata) -> list[Violation]:
    violations = []
    _, _, ext_draft, _, _, _ = _get_process_limits(part.process)
    is_pull_unknown = not part.pull_direction or part.pull_direction.lower() in ("auto", "none", "")
    pull = _parse_pull_direction(part.pull_direction)
    for face in part.faces:
        # Check external vertical face draft (Plane face and feature name doesn't contain core/hole/pocket)
        is_core = any(x in (face.sw_feature_name or "").lower() for x in ("core", "hole", "pocket", "internal"))
        dot = (face.normal_x * pull[0] + face.normal_y * pull[1] + face.normal_z * pull[2])
        if not is_core and face.draft_angle_deg is not None and face.draft_angle_deg < ext_draft and dot >= 0:
            violations.append(Violation(
                rule_id="DC-DRAFT-001",
                category=Category.DIE_CASTING.value,
                severity=Severity.WARNING,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"{face.draft_angle_deg:.2f}°",
                required_value=f"≥{ext_draft:.1f}° external draft",
                standard_reference="NADCA Guidelines",
                description=f"External face draft angle {face.draft_angle_deg:.2f}° below {ext_draft:.1f}° minimum — surface scuffing risk",
                fix_suggestion=f"Apply draft of {ext_draft:.1f}° minimum to external walls",
                solidworks_fix_path="Insert > Features > Draft > select external faces",
                unaddressed_risk_score=6,
                unaddressed_risk_reasoning="Inadequate external draft causes part scuffing, drag marks, and ejector pin marks",
                status="PENDING" if is_pull_unknown else "ACTIVE"
            ))
    return violations

@engine.register
def dc_draft_002(part: PartMetadata) -> list[Violation]:
    violations = []
    _, _, _, int_draft, _, _ = _get_process_limits(part.process)
    is_pull_unknown = not part.pull_direction or part.pull_direction.lower() in ("auto", "none", "")
    pull = _parse_pull_direction(part.pull_direction)
    for face in part.faces:
        # Check internal core draft (Cylinder or features with core/hole/pocket/internal)
        is_core = face.face_type.value == "Cylinder" or any(x in (face.sw_feature_name or "").lower() for x in ("core", "hole", "pocket", "internal"))
        dot = (face.normal_x * pull[0] + face.normal_y * pull[1] + face.normal_z * pull[2])
        if is_core and face.draft_angle_deg is not None and face.draft_angle_deg < int_draft and dot >= 0:
            violations.append(Violation(
                rule_id="DC-DRAFT-002",
                category=Category.DIE_CASTING.value,
                severity=Severity.CRITICAL,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"{face.draft_angle_deg:.2f}°",
                required_value=f"≥{int_draft:.1f}° internal core draft",
                standard_reference="NADCA Guidelines",
                description=f"Internal core draft angle {face.draft_angle_deg:.2f}° below {int_draft:.1f}° minimum — core sticking risk",
                fix_suggestion=f"Apply draft of {int_draft:.1f}° minimum to internal cores",
                solidworks_fix_path="Insert > Features > Draft > select internal core faces",
                unaddressed_risk_score=8,
                unaddressed_risk_reasoning="Casting shrinks onto cores; inadequate draft results in severe sticking, bent pins, and ejector failures",
                status="PENDING" if is_pull_unknown else "ACTIVE"
            ))
    return violations

@engine.register
def dc_corner_001(part: PartMetadata) -> list[Violation]:
    violations = []
    _, _, _, _, min_corner, _ = _get_process_limits(part.process)
    for face in part.faces:
        if face.radius_mm is not None and 0.001 < face.radius_mm < min_corner:
            violations.append(Violation(
                rule_id="DC-CORNER-001",
                category=Category.DIE_CASTING.value,
                severity=Severity.CRITICAL,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"R{face.radius_mm:.2f}mm",
                required_value=f"≥{min_corner:.1f}mm internal corner radius",
                standard_reference="NADCA Guidelines / Varroc DFM-DIE-2025",
                description=f"Internal corner radius {face.radius_mm:.2f}mm below {min_corner:.1f}mm minimum — crack and wash-out risk",
                fix_suggestion=f"Increase fillet/radius to {min_corner:.1f}mm minimum",
                solidworks_fix_path="Insert > Features > Fillet",
                unaddressed_risk_score=8,
                unaddressed_risk_reasoning="Sharp internal corners create stress concentrations leading to early die thermal fatigue cracks"
            ))
    return violations

@engine.register
def dc_poros_001(part: PartMetadata) -> list[Violation]:
    violations = []
    # wall_thickness > 6.0mm in aluminium -> WARNING (porosity)
    is_al = "al" in (part.process or "").lower() or "aluminium" in (part.process or "").lower()
    if is_al:
        for face in part.faces:
            if face.thickness_mm is not None and face.thickness_mm > 6.0:
                violations.append(Violation(
                    rule_id="DC-POROS-001",
                    category=Category.DIE_CASTING.value,
                    severity=Severity.WARNING,
                    face_ids=[face.face_id],
                    solidworks_feature_name=face.sw_feature_name,
                    measured_value=f"{face.thickness_mm:.2f}mm",
                    required_value="≤6.0mm to avoid porosity",
                    standard_reference="Varroc DFM-DIE-2025",
                    description=f"Wall thickness {face.thickness_mm:.2f}mm is greater than 6.0mm in Aluminium — severe porosity risk",
                    fix_suggestion="Reduce wall thickness or add internal core-outs",
                    solidworks_fix_path="Apply shell features to ensure wall thickness is below 5.0-6.0mm",
                    unaddressed_risk_score=7,
                    unaddressed_risk_reasoning="Heavy sections solidifying under high pressures lead to sub-surface shrinkage voids/porosity"
                ))
    return violations


# DC-UNDER-001 is now registered in injection.py with proper geometric
# undercut detection using dot-product analysis of face normals vs pull direction.
