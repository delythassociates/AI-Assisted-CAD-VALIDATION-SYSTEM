from .engine import engine
from ..core.models import PartMetadata, Violation, Severity, Category

@engine.register
def asm_gap_001(part: PartMetadata) -> list[Violation]:
    violations = []
    for gap in part.assembly_gaps:
        gap_mm = gap.get("gap_mm", 0)
        if 0 < gap_mm < 0.2:
            violations.append(Violation(
                rule_id="ASM-GAP-001",
                category=Category.ASSEMBLY.value,
                severity=Severity.WARNING,
                face_ids=[gap.get("face_a", ""), gap.get("face_b", "")],
                measured_value=f"{gap_mm:.2f}mm gap",
                required_value="≥0.2mm minimum clearance for assembly and thermal expansion",
                standard_reference="Varroc DFM-ASM-2025",
                description=f"Assembly gap {gap_mm:.2f}mm insufficient — thermal expansion causes binding in automotive range",
                fix_suggestion="Increase gap to 0.3mm or verify CTE analysis",
                solidworks_fix_path="Edit mating feature dimensions OR use Configurations for as-shipped vs in-use",
                unaddressed_risk_score=7,
                unaddressed_risk_reasoning="Thermal expansion causes interference, binding, and squeak/rattle in service"
            ))
    return violations

@engine.register
def asm_clr_001(part: PartMetadata) -> list[Violation]:
    violations = []
    for gap in part.assembly_gaps:
        gap_mm = gap.get("gap_mm", 0)
        fastener_type = gap.get("fastener_type", "")
        if "M3" in fastener_type and gap_mm < 0.8:
            violations.append(Violation(
                rule_id="ASM-CLR-001",
                category=Category.ASSEMBLY.value,
                severity=Severity.CRITICAL,
                face_ids=[gap.get("face_a", ""), gap.get("face_b", "")],
                measured_value=f"{gap_mm:.2f}mm clearance",
                required_value="≥0.8mm for M3 fastener per ISO 273",
                standard_reference="Varroc DFM-ASM-2025 / ISO 273",
                description=f"M3 fastener clearance {gap_mm:.2f}mm below 0.8mm minimum — assembly binding risk",
                fix_suggestion="Increase clearance hole to 3.8mm or use M2.5 fastener",
                solidworks_fix_path="Edit Hole Wizard feature > set clearance hole to 3.8mm",
                unaddressed_risk_score=9,
                unaddressed_risk_reasoning="Insufficient fastener clearance causes assembly binding, stripped threads, and field failures"
            ))
    return violations
