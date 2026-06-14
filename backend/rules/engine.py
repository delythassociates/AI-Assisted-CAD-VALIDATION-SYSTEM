from ..core.models import PartMetadata, Violation, Severity

# Canonical process IDs as sent by the C# add-in TaskPane
# Maps spec process IDs to their display names and rule counts
PROCESS_INFO = {
    "injection_moulding": {
        "name": "Injection Moulding",
        "description": "Thermoplastic part design for injection moulding",
        "rule_count": 3
    },
    "die_cast_al": {
        "name": "Die Casting (Aluminium)",
        "description": "Aluminium die-cast part design",
        "rule_count": 9
    },
    "die_cast_zn": {
        "name": "Die Casting (Zinc)",
        "description": "Zinc die-cast part design",
        "rule_count": 9
    },
    "die_cast_mg": {
        "name": "Die Casting (Magnesium)",
        "description": "Magnesium die-cast part design",
        "rule_count": 9
    },
}

# Category values from Violation.category → canonical process IDs
# injection.py uses Category.INJECTION_MOLDING.value = "injection_molding"
# die_casting.py uses Category.DIE_CASTING.value = "die_casting"
# We map the rule-category string to the process-family prefix for matching.
CATEGORY_TO_FAMILY = {
    "injection_molding": "injection_moulding",
    "die_casting":       "die_cast",   # matches die_cast_al / die_cast_zn / die_cast_mg
    "cnc":               "cnc",
    "assembly":          "assembly",
    "gdt":               "gdt",
}

ALWAYS_INCLUDE = {"assembly", "gdt"}


class RulesEngine:
    def __init__(self):
        self.rules = []
        self.registered_rule_ids = set()

    def register(self, rule_func):
        self.rules.append(rule_func)
        name = rule_func.__name__.upper()
        rid = name.replace("_", "-")
        self.registered_rule_ids.add(rid)
        return rule_func

    def _process_matches(self, rule_category: str, part_process: str) -> bool:
        """Return True if a rule's category is applicable for the given part process."""
        rule_cat = (rule_category or "").lower()
        proc = (part_process or "").lower()

        family = CATEGORY_TO_FAMILY.get(rule_cat)
        if family is None:
            return rule_cat in ALWAYS_INCLUDE

        if family in ALWAYS_INCLUDE:
            return True

        # Exact match OR prefix match (e.g. "die_cast" matches "die_cast_al")
        return proc == family or proc.startswith(family)

    def validate(self, part: PartMetadata) -> list[Violation]:
        violations = []
        seen = set()
        for rule in self.rules:
            result = rule(part)
            for v in result:
                if self._process_matches(v.category, part.process):
                    key = (v.rule_id, tuple(sorted(v.face_ids)))
                    if key not in seen:
                        seen.add(key)
                        violations.append(v)
        return violations

    def get_passed_rules(self, part: PartMetadata, violations: list[Violation] = None) -> list[str]:
        """Return rule IDs relevant to this part's process that produced no violation.

        If ``violations`` is passed in (already computed), avoids a second engine run.
        """
        if violations is None:
            violations = self.validate(part)
        triggered_ids = {v.rule_id for v in violations}

        proc = (part.process or "").lower()
        relevant_ids = set()
        for rid in self.registered_rule_ids:
            # Determine rule family from rule_id prefix (INJ, DC, CNC, ASM, GDT)
            prefix = rid.split("-")[0]
            if prefix == "INJ" and (proc == "injection_moulding"):
                relevant_ids.add(rid)
            elif prefix == "DC" and proc.startswith("die_cast"):
                relevant_ids.add(rid)
            elif prefix == "CNC" and proc == "cnc":
                relevant_ids.add(rid)
            elif prefix in ("ASM", "GDT"):
                relevant_ids.add(rid)

        passed = sorted(relevant_ids - triggered_ids)
        return passed

    def compute_score(self, violations: list[Violation], part: PartMetadata) -> int:
        """Score = 100 minus penalty per violation weighted by unaddressed_risk_score.

        Penalty formula:
          CRITICAL: up to -20 (scaled by risk_score/10)
          WARNING:  up to -12
          INFO:     up to -5
        Active violations only (PENDING excluded from score).
        """
        score = 100
        for v in violations:
            if getattr(v, "status", "ACTIVE") == "PENDING":
                continue
            weight = (v.unaddressed_risk_score or 5) / 10.0
            if v.severity == Severity.CRITICAL:
                score -= int(round(20 * weight))
            elif v.severity == Severity.WARNING:
                score -= int(round(12 * weight))
            else:
                score -= int(round(5 * weight))
        return max(0, min(100, score))

    def get_available_processes(self) -> list[dict]:
        return [
            {
                "id": pid,
                "name": info["name"],
                "description": info["description"],
                "rule_count": info["rule_count"]
            }
            for pid, info in PROCESS_INFO.items()
        ]


engine = RulesEngine()
