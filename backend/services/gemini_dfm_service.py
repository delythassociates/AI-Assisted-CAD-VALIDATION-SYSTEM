import os
import json
import time
import requests
import logging

logger = logging.getLogger(__name__)

# ── Friendly display names for rule IDs ──────────────────────────────────────
RULE_LABELS = {
    "INJ-WALL-001": "Wall Too Thin",
    "INJ-WALL-002": "Wall Too Thick",
    "INJ-WALL-003": "Uneven Wall Thickness",
    "INJ-DRAFT-001": "Insufficient Draft Angle",
    "INJ-DRAFT-002": "Class A Draft Too Low",
    "INJ-CORNER-001": "Sharp Internal Corner",
    "INJ-HOLE-001":  "Hole Too Small",
    "INJ-FLOW-001":  "Flow Length Too High",
    "INJ-RIB-001":   "Rib Too Thick",
    "INJ-RIB-002":   "Rib Too Tall",
    "INJ-RIB-003":   "Rib Root Radius Too Small",
    "INJ-BOSS-001":  "Boss Wall Too Thick",
    "INJ-BOSS-002":  "Boss Too Tall",
    "INJ-BOSS-003":  "Isolated Boss",
    "DC-WALL-001":   "Die-Cast Wall Too Thin",
    "DC-WALL-002":   "Die-Cast Wall Too Thick",
    "DC-WALL-003":   "Die-Cast Uneven Walls",
    "DC-DRAFT-001":  "Die-Cast Draft Too Low",
    "DC-DRAFT-002":  "Die-Cast Internal Draft Too Low",
    "DC-CORNER-001": "Die-Cast Sharp Corner",
    "DC-POROS-001":  "Die-Cast Porosity Risk",
    "DC-UNDER-001":  "Die-Cast Undercut",
    "DC-HOLE-001":   "Die-Cast Hole Too Small",
    # New rules
    "INJ-UNDERCUT-001": "Injection Undercut Detected",
    "INJ-SINK-001":     "Sink Mark Risk",
    "INJ-PARTING-001":  "Parting Line on Class A",
    "DC-UNDERCUT-001":  "Die-Cast Undercut (Geometric)",
}

PASSED_LABELS = {
    "INJ-WALL-001": "Wall thickness meets minimum",
    "INJ-WALL-002": "No excessively thick walls",
    "INJ-WALL-003": "Uniform wall thickness — good flow balance",
    "INJ-DRAFT-001": "Draft angles are sufficient",
    "INJ-DRAFT-002": "Class A faces have adequate draft",
    "INJ-CORNER-001": "Internal corners are properly radiused",
    "INJ-HOLE-001":  "Hole diameters are manufacturable",
    "INJ-FLOW-001":  "Flow length within limits",
    "INJ-RIB-001":   "Rib thickness within spec",
    "INJ-RIB-002":   "Rib height-to-thickness ratio OK",
    "INJ-RIB-003":   "Rib root radii are adequate",
    "INJ-BOSS-001":  "Boss wall thickness within spec",
    "INJ-BOSS-002":  "Boss height-to-diameter ratio OK",
    "INJ-BOSS-003":  "No isolated bosses",
    "DC-WALL-001":   "Die-cast wall thickness OK",
    "DC-WALL-002":   "No excessively thick die-cast walls",
    "DC-WALL-003":   "Die-cast wall transitions smooth",
    "DC-DRAFT-001":  "Die-cast external draft angles sufficient",
    "DC-DRAFT-002":  "Die-cast internal core draft sufficient",
    "DC-CORNER-001": "Die-cast corners properly radiused",
    "DC-POROS-001":  "No porosity risk in die-cast walls",
    "DC-UNDER-001":  "No die-cast undercuts detected",
    "DC-HOLE-001":   "Die-cast hole diameters OK",
    "GDT-POS-001":   "GD&T position tolerances reasonable",
    "ASM-GAP-001":   "Assembly gaps within tolerance",
    # New rules
    "INJ-UNDERCUT-001": "No injection undercuts detected",
    "INJ-SINK-001":     "No sink mark risk -- wall thickness uniform",
    "INJ-PARTING-001":  "No Class A faces at parting line",
    "DC-UNDERCUT-001":  "No die-cast undercuts (geometric check)",
}


def get_rule_label(rule_id: str) -> str:
    return RULE_LABELS.get(rule_id, rule_id)


def get_passed_label(rule_id: str) -> str:
    return PASSED_LABELS.get(rule_id, f"{rule_id} — passed")


def enrich_validation_with_gemini(violations: list, gnn_high_risk_faces: list, part_metadata) -> list:
    """
    Enriches DFM validation results using Gemini 2.5 Flash.

    Returns a list of enriched violation dicts.  Each dict may have
    highlight_color='GREEN' to indicate a face that passed all checks and
    deserves a positive callout in the UI.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping Gemini enrichment.")
        return []

    # Deduplicate GNN faces that are already covered by rule violations
    violated_face_ids = set()
    for v in violations:
        violated_face_ids.update(v.get("face_ids", []))

    unique_gnn_faces = [
        f for f in gnn_high_risk_faces if f.get("face_id") not in violated_face_ids
    ]

    # Build concise input payload
    input_data = {
        "process": part_metadata.process,
        "material": part_metadata.material,
        "nominal_wall_mm": part_metadata.nominal_wall_mm,
        "classification": part_metadata.classification,
        "pull_direction": part_metadata.pull_direction,
        "violations": [
            {
                "rule_id": v.get("rule_id"),
                "rule_label": get_rule_label(v.get("rule_id", "")),
                "severity": v.get("severity"),
                "face_ids": v.get("face_ids"),
                "measured_value": v.get("measured_value"),
                "required_value": v.get("required_value"),
                "description": v.get("description"),
                "fix_suggestion": v.get("fix_suggestion"),
            }
            for v in violations
        ],
        "gnn_high_risk_faces": [
            {
                "face_id": f.get("face_id"),
                "face_type": f.get("face_type"),
                "area_mm2": f.get("area_mm2"),
                "thickness_mm": f.get("thickness_mm"),
                "draft_angle_deg": f.get("draft_angle_deg"),
                "radius_mm": f.get("radius_mm"),
                "depth_mm": f.get("depth_mm"),
                "width_mm": f.get("width_mm"),
            }
            for f in unique_gnn_faces
        ],
    }

    prompt = (
        "You are a senior DFM (Design for Manufacturability) engineer reviewing a CAD part.\n"
        "Your job is to:\n"
        "  1. ENRICH each rule violation with a precise, human-readable explanation and a concrete fix.\n"
        "  2. DIAGNOSE each GNN high-risk face that has no rule violation and explain the likely manufacturing issue.\n"
        "  3. IDENTIFY up to 3 positive highlights — faces or design aspects that are well executed —\n"
        "     and represent them as GREEN entries so the designer receives encouraging feedback.\n\n"
        f"Design Context:\n{json.dumps(input_data, indent=2)}\n\n"
        "INSTRUCTIONS (follow exactly):\n"
        "• For every CRITICAL violation: highlight_color = 'RED'.\n"
        "• For every WARNING violation:  highlight_color = 'ORANGE' (GNN risks) or 'YELLOW' (rule warnings).\n"
        "• For every INFO violation:     highlight_color = 'YELLOW'.\n"
        "• For GNN faces without a rule hit: assign rule_id = 'GNN-RISK-XXX' (sequential), highlight_color = 'ORANGE'.\n"
        "• For GREEN entries (good design aspects): set rule_id = 'GOOD-DESIGN-XXX', face_id = 'none',\n"
        "  highlight_color = 'GREEN', and write a warm, specific compliment (e.g. 'Excellent wall uniformity —\n"
        "  thickness variation <5% keeps shrinkage predictable across all sections.').\n\n"
        "plain_english rules:\n"
        "  - Write in plain language a junior engineer can understand.\n"
        "  - Explain WHAT the problem is and WHY it matters for manufacturing.\n"
        "  - For GREEN entries, explain specifically what the designer did right.\n"
        "  - Max 2 sentences.\n\n"
        "fix_instruction rules:\n"
        "  - Give the exact SolidWorks action (menu path, feature name, or sketch edit).\n"
        "  - Include the target dimension value.\n"
        "  - For GREEN entries, write 'No action required — maintain this standard.'.\n\n"
        "Output ONLY a valid JSON array. No markdown, no preamble, no trailing text.\n"
        "Each object MUST contain exactly these fields:\n"
        "  rule_id           (string)\n"
        "  face_id           (string — first face_id from the violation, or 'none' for GREEN entries)\n"
        "  current_value_mm  (float or null)\n"
        "  minimum_required_mm (float or null)\n"
        "  optimal_value_mm  (float or null)\n"
        "  fix_delta_mm      (float or null: minimum_required_mm - current_value_mm)\n"
        "  plain_english     (string)\n"
        "  fix_instruction   (string)\n"
        "  highlight_color   ('RED' | 'ORANGE' | 'YELLOW' | 'GREEN')\n"
        "  confidence        (float 0.0-1.0: how confident you are in this assessment)\n"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=8.0)
        response.raise_for_status()
        resp_json = response.json()
        raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Strip markdown fences if model ignores responseMimeType
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
            raw_text = "\n".join(lines).strip()

        parsed = json.loads(raw_text)
        logger.info(
            f"Gemini enrichment OK — {len(parsed)} items "
            f"({sum(1 for x in parsed if x.get('highlight_color') == 'GREEN')} GREEN, "
            f"{sum(1 for x in parsed if x.get('highlight_color') == 'RED')} RED)"
        )
        return parsed
    except Exception as exc:
        logger.error(f"Gemini enrichment failed: {exc}")
        return []

    return []
