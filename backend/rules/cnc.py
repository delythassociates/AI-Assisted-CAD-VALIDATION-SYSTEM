from .engine import engine
from ..core.models import PartMetadata, Violation, Severity, Category

@engine.register
def cnc_depth_001(part: PartMetadata) -> list[Violation]:
    violations = []
    for face in part.faces:
        if face.depth_mm is not None and face.width_mm is not None and face.width_mm > 0:
            ratio = face.depth_mm / face.width_mm
            if ratio > 4.0:
                violations.append(Violation(
                    rule_id="CNC-DEPTH-001",
                    category=Category.CNC.value,
                    severity=Severity.CRITICAL,
                    face_ids=[face.face_id],
                    solidworks_feature_name=face.sw_feature_name,
                    measured_value=f"Depth/width ratio {ratio:.1f}",
                    required_value="≤4:1 for standard tooling",
                    standard_reference="Varroc DFM-CNC-2025 / ISO 286",
                    description=f"Pocket depth/width ratio {ratio:.1f} exceeds 4:1 — tool deflection and vibration risk",
                    fix_suggestion="Increase pocket width, reduce depth, or use variable helix tooling",
                    solidworks_fix_path="Edit Cut-Extrude depth OR add secondary operation note",
                    unaddressed_risk_score=9,
                    unaddressed_risk_reasoning="Tool chatter and deflection cause poor surface finish, tolerance failure, and tool breakage"
                ))
    return violations

@engine.register
def cnc_radius_001(part: PartMetadata) -> list[Violation]:
    violations = []
    for face in part.faces:
        if face.radius_mm is not None and face.radius_mm < 2.0 and face.radius_mm > 0:
            violations.append(Violation(
                rule_id="CNC-RADIUS-001",
                category=Category.CNC.value,
                severity=Severity.WARNING,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"{face.radius_mm:.2f}mm",
                required_value="≥2.0mm internal corner radius for standard end mills",
                standard_reference="Varroc DFM-CNC-2025",
                description=f"Internal corner radius {face.radius_mm:.2f}mm below 2.0mm — requires special tooling and slows cycle",
                fix_suggestion="Increase internal corner radius to at least 2.0mm (preferably 4.0mm)",
                solidworks_fix_path="Insert > Features > Fillet > select internal edges > set radius to 4.0mm",
                unaddressed_risk_score=6,
                unaddressed_risk_reasoning="Small corner radii require special ball end mills, increase cycle time, and cause tool deflection"
            ))
    return violations
