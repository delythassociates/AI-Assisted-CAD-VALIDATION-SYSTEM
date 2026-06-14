from fastapi import APIRouter
from ..core.models import Violation, FixSuggestion, PartMetadata

router = APIRouter()

FIX_TEMPLATES = {
    "INJ-WALL-001": {
        "root_cause": "Wall thickness below minimum required for material flow — material cools before filling cavity completely, creating weak knit lines.",
        "immediate_fix": {
            "description": "Increase wall thickness on affected faces to minimum 0.9mm",
            "solidworks_steps": [
                "Right-click the Boss-Extrude/Shell feature in FeatureManager Design Tree",
                "Select Edit Feature",
                "Change thickness dimension to 1.0mm",
                "Click OK — rebuild model (Ctrl+B)"
            ],
            "solidworks_menu_path": "FeatureManager > right-click feature > Edit Feature",
            "solidworks_api_call": "swDoc.Parameter('D1@Sketch1').SystemValue = 0.001",
            "estimated_time_minutes": 10
        },
        "alternative_fixes": [
            {
                "description": "Add structural ribs at 60% of wall thickness instead of increasing entire wall",
                "trade_off": "Adds mould complexity but maintains lighter weight",
                "approval_required": True
            }
        ],
        "downstream_impacts": ["Increase in part weight", "May affect cooling time", "Check for sink marks on opposing face"],
        "verification_method": "Use SolidWorks Measure tool (Tools > Evaluate > Measure) to confirm wall thickness ≥ 0.9mm on affected face"
    },
    "INJ-DRAFT-001": {
        "root_cause": "Vertical walls perpendicular to mould opening direction prevent ejection — vacuum lock and part deformation on extraction.",
        "immediate_fix": {
            "description": "Apply 1.5° draft angle on all faces parallel to pull direction",
            "solidworks_steps": [
                "Insert > Features > Draft",
                "Select the neutral plane (mould parting surface)",
                "Select faces to add draft to",
                "Set angle to 1.5°",
                "Click OK"
            ],
            "solidworks_menu_path": "Insert > Features > Draft",
            "solidworks_api_call": "swDoc.InsertDraft(...)",
            "estimated_time_minutes": 15
        },
        "alternative_fixes": [
            {
                "description": "Use variable draft: 1° on textured surfaces, 3° on smooth surfaces",
                "trade_off": "More complex mould design but better surface finish",
                "approval_required": False
            }
        ],
        "downstream_impacts": ["Minor dimensional change on drafted faces", "Check assembly clearance"],
        "verification_method": "Use Draft Analysis tool (View > Display > Draft Analysis) — all faces should show ≥1.5°"
    },
    "DIE-WALL-001": {
        "root_cause": "Aluminium die casting requires minimum 2.0mm wall to ensure complete die fill and avoid cold shuts.",
        "immediate_fix": {
            "description": "Increase wall to 2.5mm minimum",
            "solidworks_steps": [
                "Edit the Boss-Extrude feature",
                "Increase extrusion thickness to 2.5mm",
                "Rebuild"
            ],
            "solidworks_menu_path": "FeatureManager > Edit Feature > change dimension",
            "estimated_time_minutes": 5
        },
        "alternative_fixes": [
            {
                "description": "Change material to higher-fluidity alloy (e.g., A380)",
                "trade_off": "Material cost increase ~15%",
                "approval_required": True
            }
        ],
        "downstream_impacts": ["Weight increase", "Check if holes still meet tolerance"],
        "verification_method": "Section view in SW and measure wall thickness"
    },
    "CNC-DEPTH-001": {
        "root_cause": "Deep narrow pockets require long reach tools that deflect under cutting forces, causing taper and poor surface finish.",
        "immediate_fix": {
            "description": "Reduce pocket depth or increase width to achieve ≤6:1 ratio",
            "solidworks_steps": [
                "Edit Cut-Extrude feature for the pocket",
                "Reduce depth or increase width in sketch",
                "Rebuild"
            ],
            "solidworks_menu_path": "FeatureManager > right-click Cut-Extrude > Edit Sketch",
            "estimated_time_minutes": 20
        },
        "alternative_fixes": [
            {
                "description": "Add relief at pocket corners for stepped machining",
                "trade_off": "Adds manufacturing step but allows deeper pockets",
                "approval_required": True
            }
        ],
        "downstream_impacts": ["Reduced pocket capacity", "May affect part strength"],
        "verification_method": "Calculate new depth/width ratio from SW measure"
    },
    "DIE-DRAFT-001": {
        "root_cause": "Insufficient draft in die casting prevents clean ejection — aluminium shrinks onto cores, requiring excessive force.",
        "immediate_fix": {
            "description": "Apply 3° draft on internal walls, 2.5° on external",
            "solidworks_steps": [
                "Insert > Features > Draft",
                "Select parting line or neutral plane",
                "Set angle to 3° for internal faces",
                "Apply"
            ],
            "solidworks_menu_path": "Insert > Features > Draft",
            "estimated_time_minutes": 15
        },
        "alternative_fixes": [], "downstream_impacts": ["Check assembly with mating parts"],
        "verification_method": "Draft analysis in SW — all surfaces ≥2°"
    }
}

DEFAULT_FIX = {
    "root_cause": "Design violates standard manufacturing constraint. See rule reference for details.",
    "immediate_fix": {
        "description": "Review rule requirements and modify feature geometry accordingly",
        "solidworks_steps": ["Identify affected feature in FeatureManager", "Right-click and Edit Feature", "Adjust dimension to meet requirement", "Rebuild and re-validate"],
        "solidworks_menu_path": "FeatureManager > Edit Feature",
        "estimated_time_minutes": 30
    },
    "alternative_fixes": [], "downstream_impacts": ["Requires re-validation"],
    "verification_method": "Re-run Eureka validation to confirm fix"
}

@router.post("", response_model=FixSuggestion)
async def get_fix_suggestion(violation: Violation, part: PartMetadata):
    template = FIX_TEMPLATES.get(violation.rule_id, DEFAULT_FIX)
    fix = FixSuggestion(
        rule_id=violation.rule_id,
        root_cause=template["root_cause"],
        immediate_fix=template["immediate_fix"],
        alternative_fixes=template.get("alternative_fixes", []),
        downstream_impacts=template.get("downstream_impacts", []),
        verification_method=template.get("verification_method", "Re-run validation"),
        estimated_time_minutes=template["immediate_fix"].get("estimated_time_minutes", 30)
    )
    return fix
