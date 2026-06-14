from .engine import engine
from ..core.models import PartMetadata, Violation, Severity, Category

@engine.register
def gdt_flat_001(part: PartMetadata) -> list[Violation]:
    violations = []
    for face in part.faces:
        if face.area_mm2 > 2000 and face.face_type.value == "Plane":
            violations.append(Violation(
                rule_id="GDT-FLAT-001",
                category=Category.GDT.value,
                severity=Severity.INFO,
                face_ids=[face.face_id],
                solidworks_feature_name=face.sw_feature_name,
                measured_value=f"Face area {face.area_mm2:.0f}mm²",
                required_value="Planar face > 2000mm² — specify flatness tolerance",
                standard_reference="ISO 1101 / ASME Y14.5 / Varroc DFM-GDT-2025",
                description=f"Planar face ({face.area_mm2:.0f}mm²) over 2000mm² needs explicit flatness callout for sealing and fit",
                fix_suggestion="Add flatness tolerance of 0.05mm or tighter depending on sealing requirement",
                solidworks_fix_path="Insert > DimXpert > Geometric Tolerance > Flatness > select face > set value",
                unaddressed_risk_score=5,
                unaddressed_risk_reasoning="Uncontrolled flatness on sealing faces leads to leak paths and joint failure"
            ))
    return violations

@engine.register
def gdt_cyl_001(part: PartMetadata) -> list[Violation]:
    violations = []
    for face in part.faces:
        if face.face_type.value in ("Cylinder",) and face.radius_mm is not None:
            if face.depth_mm is not None and face.depth_mm / (face.radius_mm * 2) > 2.0:
                violations.append(Violation(
                    rule_id="GDT-CYL-001",
                    category=Category.GDT.value,
                    severity=Severity.INFO,
                    face_ids=[face.face_id],
                    measured_value=f"L/D ratio {face.depth_mm/(face.radius_mm*2):.1f}",
                    required_value="Specify cylindricity for L/D > 2",
                    standard_reference="ISO 1101 / ASME Y14.5 / Varroc DFM-GDT-2025",
                    description=f"Cylindrical feature with L/D > 2 should have cylindricity tolerance for consistent fit",
                    fix_suggestion="Add cylindricity tolerance of 0.03mm",
                    solidworks_fix_path="Insert > DimXpert > Geometric Tolerance > Cylindricity",
                    unaddressed_risk_score=4,
                    unaddressed_risk_reasoning="Without cylindricity control, mating part fit is inconsistent and sealing compromised"
                ))
    return violations

@engine.register
def gdt_pos_001(part: PartMetadata) -> list[Violation]:
    violations = []
    hole_count = sum(1 for f in part.faces if f.face_type.value == "Cylinder" and f.radius_mm is not None and f.radius_mm < 8.0)
    if hole_count >= 2:
        violations.append(Violation(
            rule_id="GDT-POS-001",
            category=Category.GDT.value,
            severity=Severity.INFO,
            face_ids=[],
            measured_value=f"{hole_count} holes detected",
            required_value="True position tolerance for hole patterns (≥2 holes)",
            standard_reference="ISO 1101 / ASME Y14.5 / Varroc DFM-GDT-2025",
            description=f"Part has {hole_count} holes — true position tolerance required for pattern control and assembly",
            fix_suggestion="Add true position tolerance referencing datums A, B, C",
            solidworks_fix_path="Insert > DimXpert > Geometric Tolerance > Position > select hole pattern",
            unaddressed_risk_score=4,
            unaddressed_risk_reasoning="Without true position, bolt-hole misalignment causes assembly failures and rework"
        ))
    return violations
