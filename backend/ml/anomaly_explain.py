import os, json, time, requests, logging
from ..core.models import PartMetadata

logger = logging.getLogger(__name__)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_AVAILABLE = bool(GEMINI_KEY)

FEATURE_NAMES = {
    0: "face_type", 1: "thickness_mm", 2: "radius_mm", 3: "area_mm2",
    4: "depth_mm", 5: "width_mm", 6: "curvature_min", 7: "curvature_max",
    8: "draft_angle_deg", 9: "parent_wall_thickness_mm",
    10: "centroid_dist", 11: "aspect_ratio"
}

def _build_gemini_prompt(gnn_score: float, part: PartMetadata, heuristic: dict) -> str:
    faces_summary = []
    for f in part.faces[:10]:
        face_id = getattr(f, "face_id", "Unknown")
        face_type_val = "Unknown"
        if hasattr(f, "face_type") and f.face_type:
            face_type_val = f.face_type.value if hasattr(f.face_type, "value") else str(f.face_type)
        
        thickness = f"{f.thickness_mm:.2f}mm" if getattr(f, "thickness_mm", None) is not None else "N/A"
        draft = f"{f.draft_angle_deg:.2f}°" if getattr(f, "draft_angle_deg", None) is not None else "N/A"
        radius = f"{f.radius_mm:.2f}mm" if getattr(f, "radius_mm", None) is not None else "N/A"
        depth = f"{f.depth_mm:.2f}mm" if getattr(f, "depth_mm", None) is not None else "N/A"
        width = f"{f.width_mm:.2f}mm" if getattr(f, "width_mm", None) is not None else "N/A"
        
        faces_summary.append(
            f"Face {face_id}: type={face_type_val}, thickness={thickness}, draft={draft}, radius={radius}, depth={depth}, width={width}"
        )

    v_summary = heuristic.get("interaction_description", "No issues")
    fm = heuristic.get("predicted_failure_modes", [])
    
    proc = (part.process or "").lower()
    if "inject" in proc:
        process_friendly = "Injection Moulding"
    elif "cast" in proc:
        process_friendly = "Die Casting"
    elif "cnc" in proc:
        process_friendly = "CNC Machining"
    else:
        process_friendly = part.process or "General Manufacturing"

    return (
        f"DFM anomaly detected for a CAD part designed for {process_friendly}.\n"
        f"Part Name: {part.filename or 'Unknown'}\n"
        f"Process: {process_friendly}\n"
        f"Material: {part.material or 'Unknown'}\n"
        f"GNN risk score: {gnn_score:.4f}\n"
        f"Total face count: {len(part.faces)}\n"
        f"Key Faces (first 10):\n" + "\n".join(faces_summary) + "\n"
        f"Rule-based analysis summary: {v_summary}\n"
        f"Predicted failure modes: {', '.join(fm) if fm else 'None'}\n\n"
        f"Explain the root cause of this geometric anomaly and recommend the top fix tailored specifically for the {process_friendly} process.\n"
        f"Respond in exactly 3 concise, clear bullet points. Each bullet point should be professional and easy to understand for a designer."
    )

def explain_anomaly(gnn_score: float, part: PartMetadata) -> dict:
    faces = part.faces
    thin_faces = [f for f in faces if f.thickness_mm is not None and f.thickness_mm < 1.5]
    low_draft = [f for f in faces if f.draft_angle_deg is not None and f.draft_angle_deg < 2.0]
    deep_ribs = [f for f in faces if f.depth_mm is not None and f.depth_mm > 5.0 and "Rib" in (f.sw_feature_name or "")]
    thick_walls = [f for f in faces if f.thickness_mm is not None and f.thickness_mm > 3.0]
    deep_cores = [f for f in faces if f.depth_mm is not None and f.width_mm is not None and f.width_mm > 0 and f.depth_mm / f.width_mm > 2.0]
    large_thin = [f for f in faces if f.thickness_mm is not None and f.area_mm2 is not None and f.thickness_mm < 1.5 and f.area_mm2 > 3000]

    combinations = []
    combos_found = []

    if thin_faces and low_draft:
        combos_found.append({
            "feature_1": {"name": "wall_thickness_mm", "value": thin_faces[0].thickness_mm, "status": "PASS"},
            "feature_2": {"name": "draft_angle_deg", "value": low_draft[0].draft_angle_deg, "status": "PASS"},
            "combined_risk": f"{thin_faces[0].thickness_mm:.1f}mm wall + {low_draft[0].draft_angle_deg:.1f}° draft creates sink mark probability >{gnn_score*100:.0f}%"
        })
        combinations.append(f"thin wall ({thin_faces[0].thickness_mm:.1f}mm) + low draft ({low_draft[0].draft_angle_deg:.1f}°)")

    if thin_faces and deep_ribs:
        combos_found.append({
            "feature_1": {"name": "wall_thickness_mm", "value": thin_faces[0].thickness_mm, "status": "PASS"},
            "feature_2": {"name": "rib_height_mm", "value": deep_ribs[0].depth_mm, "status": "PASS"},
            "combined_risk": f"{thin_faces[0].thickness_mm:.1f}mm wall + {deep_ribs[0].depth_mm:.1f}mm rib — sink mark and fill imbalance"
        })
        combinations.append(f"thin wall ({thin_faces[0].thickness_mm:.1f}mm) + deep rib ({deep_ribs[0].depth_mm:.1f}mm)")

    if thick_walls and low_draft:
        combos_found.append({
            "feature_1": {"name": "wall_thickness_mm", "value": thick_walls[0].thickness_mm, "status": "PASS"},
            "feature_2": {"name": "draft_angle_deg", "value": low_draft[0].draft_angle_deg, "status": "PASS"},
            "combined_risk": f"{thick_walls[0].thickness_mm:.1f}mm wall + {low_draft[0].draft_angle_deg:.1f}° draft — extended cooling time and ejection risk"
        })
        combinations.append(f"thick wall ({thick_walls[0].thickness_mm:.1f}mm) + low draft ({low_draft[0].draft_angle_deg:.1f}°)")

    if thick_walls and deep_cores:
        combos_found.append({
            "feature_1": {"name": "wall_thickness_mm", "value": thick_walls[0].thickness_mm, "status": "PASS"},
            "feature_2": {"name": "core_depth_width_ratio", "value": round(deep_cores[0].depth_mm / deep_cores[0].width_mm, 1) if deep_cores[0].width_mm > 0 else 0, "status": "PASS"},
            "combined_risk": f"{thick_walls[0].thickness_mm:.1f}mm wall + slender core — uneven cooling and shrinkage porosity"
        })
        combinations.append(f"thick wall ({thick_walls[0].thickness_mm:.1f}mm) + slender core")

    if large_thin:
        combos_found.append({
            "feature_1": {"name": "wall_thickness_mm", "value": large_thin[0].thickness_mm, "status": "PASS"},
            "feature_2": {"name": "area_mm2", "value": large_thin[0].area_mm2, "status": "PASS"},
            "combined_risk": f"{large_thin[0].thickness_mm:.1f}mm wall over {large_thin[0].area_mm2:.0f}mm² — warpage and sink mark risk"
        })
        combinations.append(f"thin ({large_thin[0].thickness_mm:.1f}mm) large-area ({large_thin[0].area_mm2:.0f}mm²) face")

    involved_face_ids = list(set(
        f.face_id for f in (thin_faces[:2] + low_draft[:2] + deep_ribs[:2] + thick_walls[:2] + deep_cores[:2] + large_thin[:2])
    ))
    involved_sw_features = list(set(
        f.sw_feature_name for f in (thin_faces[:3] + low_draft[:3] + deep_ribs[:3] + thick_walls[:3] + deep_cores[:3] + large_thin[:3])
        if f.sw_feature_name
    ))

    failure_modes = []
    if thin_faces and low_draft:
        failure_modes.append("sink_mark")
    if thick_walls and deep_cores:
        failure_modes.append("shrinkage_porosity")
    if large_thin:
        failure_modes.append("warpage")
    if not failure_modes:
        failure_modes.append("multi_feature_interaction")

    result = {
        "gnn_risk_score": round(gnn_score, 4),
        "individual_rules_passed": True,
        "multi_feature_interaction_detected": len(combos_found) > 0,
        "interaction_description": f"Multiple features combine to create risk: {'; '.join(combinations)}" if combinations else "No clear multi-feature interaction identified",
        "involved_face_ids": involved_face_ids,
        "involved_sw_features": involved_sw_features,
        "feature_combination": combos_found[0] if combos_found else {},
        "all_combinations": combos_found,
        "predicted_failure_modes": failure_modes,
        "suggested_rule_ids": ["INJ-COMBO-001", "INJ-COMBO-002"],
        "suggested_new_rule_description": "Multi-feature interaction: wall thickness, draft angle, rib height, and core geometry combine to create manufacturability risk",
        "confidence": round(min(gnn_score, 0.95), 2),
        "engineer_action": "REVIEW_REQUIRED",
        "gemini_explanation": None
    }

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        prompt = _build_gemini_prompt(gnn_score, part, result)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
            },
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=8.0)
            response.raise_for_status()
            resp_json = response.json()
            raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            result["gemini_explanation"] = raw_text
        except Exception as e:
            logger.error(f"Gemini explain_anomaly failed: {e}")
            result["gemini_explanation"] = "Analysis unavailable — GNN explanation timed out."
    else:
        result["gemini_explanation"] = "GEMINI_API_KEY not configured"

    return result
