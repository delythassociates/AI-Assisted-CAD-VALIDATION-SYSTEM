import logging
import traceback
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from ..core.models import PartMetadata, Violation, ValidationResult, Severity
from ..rules.engine import engine
from ..services import gnn_engine
from ..services.gemini_dfm_service import (
    enrich_validation_with_gemini,
    get_passed_label,
    get_rule_label,
)
from ..ml.anomaly_explain import explain_anomaly
from .router_report import generate_gemini_report_narrative
from ..utils.gemini_utils import call_gemini_with_timeout

import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()


def _save_validation_yaml(part: PartMetadata) -> str | None:
    import yaml
    import os
    import time
    try:
        from ..ml.feedback_store import DB_PATH
        db_dir = os.path.dirname(DB_PATH)
        parts_dir = os.path.join(db_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        
        safe_name = "".join([c if c.isalnum() or c in "-_" else "_" for c in part.filename])
        if not safe_name:
            safe_name = "part"
        filename = f"{safe_name}_{int(time.time() * 1000)}.yml"
        filepath = os.path.join(parts_dir, filename)
        
        import json
        part_json = part.model_dump_json() if hasattr(part, "model_dump_json") else part.json()
        part_dict = json.loads(part_json)
        with open(filepath, "w") as f:
            yaml.safe_dump(part_dict, f)
            
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save validation yaml: {e}")
        return None


def _inject_gnn_face_violations(violations: list[Violation], face_scores: dict, part: PartMetadata):
    """Surface GNN high/medium-risk faces as violations when rules did not fire."""
    covered = set()
    for v in violations:
        if v.status == "PENDING":
            continue
        for fid in (v.face_ids or []):
            covered.add(fid)

    for face in part.faces:
        fid = face.face_id
        if fid in covered:
            continue
        score = face_scores.get(fid, 0.0)
        if score <= 0.3:
            continue
        sw_feat = face.sw_feature_name or ""
        if score > 0.6:
            violations.append(Violation(
                rule_id="GNN-RISK-001",
                category=part.process,
                severity=Severity.WARNING,
                face_ids=[fid],
                solidworks_feature_name=sw_feat,
                description=f"GNN neural risk {score:.0%} on face - geometry pattern matches known defect profiles",
                fix_suggestion="Review face geometry; check wall transitions, draft, and corner radii",
                plain_english=f"AI detected elevated manufacturability risk ({score:.0%}) on this face. "
                               "The geometry shape resembles features that commonly cause defects.",
                fix_instruction="Inspect the highlighted face in SolidWorks. Check wall thickness, "
                                "draft angle, and any sharp transitions near this face.",
                highlight_color="ORANGE",
                unaddressed_risk_score=int(min(10, score * 10)),
            ))
        else:
            violations.append(Violation(
                rule_id="GNN-WATCH-001",
                category=part.process,
                severity=Severity.INFO,
                face_ids=[fid],
                solidworks_feature_name=sw_feat,
                description=f"GNN watch-zone risk {score:.0%} - monitor during design review",
                fix_suggestion="Consider minor geometry refinement if adjacent to critical features",
                plain_english=f"AI flagged this face for monitoring ({score:.0%} risk score). "
                               "Low concern now but worth reviewing alongside neighbouring features.",
                fix_instruction="Verify wall thickness and draft are within nominal targets for this region.",
                highlight_color="YELLOW",
                unaddressed_risk_score=int(min(6, score * 10)),
            ))


def _compute_face_health(violations: list[Violation], face_scores: dict, total_faces: int) -> dict:
    """Classify each face into critical / at_risk / watch / clean for UI legend."""
    face_tier = {}

    for v in violations:
        if v.status == "PENDING":
            continue
        hc = (v.highlight_color or "").upper()
        tier = 0
        if hc == "RED" or v.severity == Severity.CRITICAL:
            tier = 4
        elif hc == "ORANGE" or (v.rule_id or "").startswith("GNN-RISK"):
            tier = 3
        elif hc == "YELLOW" or v.severity == Severity.WARNING:
            tier = 2
        elif hc == "GREEN":
            tier = 1
        for fid in (v.face_ids or []):
            face_tier[fid] = max(face_tier.get(fid, 0), tier)

    for fid, score in face_scores.items():
        if fid in face_tier:
            continue
        if score > 0.6:
            face_tier[fid] = 3
        elif score > 0.3:
            face_tier[fid] = 2
        else:
            face_tier[fid] = 1

    critical = sum(1 for t in face_tier.values() if t >= 4)
    at_risk  = sum(1 for t in face_tier.values() if t == 3)
    watch    = sum(1 for t in face_tier.values() if t == 2)
    known    = len(face_tier)
    clean    = max(0, total_faces - known) + sum(1 for t in face_tier.values() if t <= 1)

    return {
        "critical": critical,
        "at_risk":  at_risk,
        "watch":    watch,
        "clean":    clean,
        "total":    total_faces,
    }


def _build_passed_checks(passed_rule_ids: list[str], gemini_good_items: list[dict]) -> list[str]:
    """
    Build a human-friendly list of passed checks for the UI.
    Combines rules-engine passed rules with Gemini GREEN items.
    """
    seen = set()
    result = []

    # Gemini GREEN items come first (most specific)
    for item in gemini_good_items:
        label = item.get("plain_english", "").strip()
        if label and label not in seen:
            seen.add(label)
            result.append(f"✓ {label}")

    # Rules-engine passed checks (use friendly labels)
    for rid in passed_rule_ids:
        label = get_passed_label(rid)
        key = rid
        if key not in seen:
            seen.add(key)
            result.append(f"✓ {label}")

    return result


@router.post("")
async def validate_part(part: PartMetadata):
    # 1. Input validation
    if len(part.faces) > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Too many faces ({len(part.faces)}). Maximum is 500 faces."},
        )
    if not part.process:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Manufacturing process is required."},
        )
    if not part.material:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Material is required."},
        )
    if part.nominal_wall_mm is None or part.nominal_wall_mm <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Nominal wall thickness must be a positive number."},
        )

    try:
        # Normalize legacy process aliases
        _process_aliases = {
            "injection_molding":  "injection_moulding",
            "die_casting":        "die_cast_al",
            "die_casting_al":     "die_cast_al",
            "die_casting_zn":     "die_cast_zn",
            "die_casting_mg":     "die_cast_mg",
        }
        if part.process in _process_aliases:
            part.process = _process_aliases[part.process]

        # 2. Rules engine
        violations = engine.validate(part)
        passed_rule_ids = engine.get_passed_rules(part, violations=violations)
        score = engine.compute_score(violations, part)

        violations_dicts = [v.model_dump() for v in violations]

        # 3. GNN inference
        gnn_score = 0.0
        face_scores = {}
        gnn_anomaly_data = {
            "gnn_risk_score": 0.0,
            "face_scores": {},
            "gemini_explanation": None,
        }
        if gnn_engine.available and len(part.faces) > 0:
            try:
                # Use hybrid (GNN+XGBoost) or pure GNN based on inference_mode
                if gnn_engine.inference_mode == "hybrid":
                    gnn_result = gnn_engine.predict_hybrid(part)
                else:
                    gnn_result = gnn_engine.predict(part)
                gnn_score   = gnn_result.get("part_risk_score", 0.0)
                face_scores = gnn_result.get("face_scores", {})

                # Flywheel: count every inference for total_parts_analyzed metric
                try:
                    from ..ml.feedback_store import log_inference
                    _pid = part.solidworks_part_number or part.filename or "unknown"
                    log_inference(_pid)
                except Exception:
                    pass
                
                gnn_anomaly_data["gnn_risk_score"] = round(gnn_score, 4)
                gnn_anomaly_data["face_scores"] = face_scores
                
            except Exception as exc:
                logger.error(f"GNN inference failed: {exc}")

        # High-risk faces for Gemini context
        gnn_high_risk_faces = [
            face.model_dump()
            for face in part.faces
            if face_scores.get(face.face_id, 0.0) > 0.6
        ]

        # 4. Gemini Parallel Calls (enrichment + GNN explanation)
        gemini_enriched = False
        gemini_good_items: list[dict] = []
        enriched_data = []

        tasks = []
        # Task 1: Enrichment
        tasks.append(call_gemini_with_timeout(
            enrich_validation_with_gemini,
            violations_dicts,
            gnn_high_risk_faces,
            part,
            timeout_seconds=8.0,
            fallback=[]
        ))

        # Task 2: GNN explanation (if score > 0.3)
        explain_task_idx = -1
        if gnn_score > 0.3:
            explain_task_idx = len(tasks)
            tasks.append(call_gemini_with_timeout(
                explain_anomaly,
                gnn_score,
                part,
                timeout_seconds=8.0,
                fallback={
                    "gnn_risk_score": round(gnn_score, 4),
                    "individual_rules_passed": True,
                    "multi_feature_interaction_detected": False,
                    "interaction_description": "Analysis unavailable — GNN explanation timed out.",
                    "involved_face_ids": [],
                    "involved_sw_features": [],
                    "feature_combination": {},
                    "all_combinations": [],
                    "predicted_failure_modes": [],
                    "suggested_rule_ids": ["GNN-ANOMALY-001"],
                    "suggested_new_rule_description": "Analysis unavailable — GNN explanation timed out.",
                    "confidence": 0.5,
                    "engineer_action": "REVIEW_REQUIRED",
                    "gemini_explanation": "Analysis unavailable — GNN explanation timed out."
                }
            ))

        try:
            results = await asyncio.gather(*tasks)
            enriched_data = results[0]
            gemini_enriched = bool(enriched_data)

            if explain_task_idx != -1:
                explain_res = results[explain_task_idx]
                for k, v in explain_res.items():
                    if k != "face_scores":
                        gnn_anomaly_data[k] = v
        except Exception as exc:
            logger.error(f"Parallel Gemini execution failed: {exc}")

        try:

            # Separate GREEN (positive) items from problem items
            problem_items = [x for x in enriched_data if x.get("highlight_color", "") != "GREEN"]
            gemini_good_items = [x for x in enriched_data if x.get("highlight_color", "") == "GREEN"]

            # Build lookup: (rule_id, face_id) → enriched item
            enriched_map = {}
            for item in problem_items:
                rid = item.get("rule_id")
                fid = item.get("face_id")
                if rid and fid:
                    enriched_map[(rid, fid)] = item

            # Merge enrichment into existing violations
            for v in violations:
                primary_face = v.face_ids[0] if v.face_ids else None
                key = (v.rule_id, primary_face)
                if key in enriched_map:
                    item = enriched_map[key]
                    v.current_value_mm    = item.get("current_value_mm")
                    v.minimum_required_mm = item.get("minimum_required_mm")
                    v.optimal_value_mm    = item.get("optimal_value_mm")
                    v.fix_delta_mm        = item.get("fix_delta_mm")
                    v.plain_english       = item.get("plain_english") or v.description
                    v.fix_instruction     = item.get("fix_instruction") or v.fix_suggestion
                    v.highlight_color     = item.get("highlight_color") or (
                        "RED" if v.severity == Severity.CRITICAL else "YELLOW"
                    )
                else:
                    # Default colouring when Gemini didn't cover this violation
                    v.highlight_color = "RED" if v.severity == Severity.CRITICAL else "YELLOW"
                    v.plain_english   = v.description
                    v.fix_instruction = v.fix_suggestion

            # Append new GNN-only violations surfaced by Gemini
            existing_keys = {
                (v.rule_id, v.face_ids[0] if v.face_ids else None)
                for v in violations
            }
            for item in problem_items:
                rid = item.get("rule_id", "")
                fid = item.get("face_id", "")
                if rid.startswith("GNN-") and (rid, fid) not in existing_keys:
                    sw_feat = next(
                        (f.sw_feature_name for f in part.faces if f.face_id == fid), ""
                    )
                    violations.append(Violation(
                        rule_id=rid,
                        category=part.process,
                        severity=Severity.WARNING,
                        face_ids=[fid] if fid else [],
                        solidworks_feature_name=sw_feat,
                        description=item.get("plain_english", "GNN anomaly detected"),
                        fix_suggestion=item.get("fix_instruction", "Review geometry"),
                        current_value_mm=item.get("current_value_mm"),
                        minimum_required_mm=item.get("minimum_required_mm"),
                        optimal_value_mm=item.get("optimal_value_mm"),
                        fix_delta_mm=item.get("fix_delta_mm"),
                        plain_english=item.get("plain_english", ""),
                        fix_instruction=item.get("fix_instruction", ""),
                        highlight_color="ORANGE",
                    ))

        except Exception as gem_ex:
            logger.error(f"Gemini integration error: {gem_ex} — using rules-only fallback.")
            gemini_enriched = False
            for v in violations:
                v.highlight_color = "RED" if v.severity == Severity.CRITICAL else "YELLOW"
                v.plain_english   = v.description
                v.fix_instruction = v.fix_suggestion

        # 5. Inject remaining GNN-only face violations not yet covered
        if gnn_score > 0.6:
            involved_faces = gnn_anomaly_data.get("involved_face_ids", [])
            if not involved_faces:
                involved_faces = [fid for fid, sc in face_scores.items() if sc > 0.6]
            
            gemini_explanation = gnn_anomaly_data.get("gemini_explanation") or "GNN neural geometric anomaly detected."
            violations.append(Violation(
                rule_id="GNN-ANOMALY-001",
                category=part.process,
                severity=Severity.WARNING,
                face_ids=involved_faces,
                solidworks_feature_name="",
                description="GNN Geometric Anomaly Detected",
                fix_suggestion="Review AI anomaly recommendations to adjust draft, thickness, or fillets.",
                plain_english=gemini_explanation,
                fix_instruction="Inspect the highlighted faces in SolidWorks to resolve the combination of thin sections and low draft.",
                highlight_color="ORANGE",
                unaddressed_risk_score=int(min(10, gnn_score * 10)),
            ))

        if face_scores:
            _inject_gnn_face_violations(violations, face_scores, part)

        # Recalculate score to include GNN violations
        score = engine.compute_score(violations, part)

        face_health = _compute_face_health(violations, face_scores, len(part.faces))

        # 6. Build human-friendly passed checks list
        passed_checks_friendly = _build_passed_checks(passed_rule_ids, gemini_good_items)

        # 7. Assemble final result
        active = [v for v in violations if getattr(v, "status", "ACTIVE") != "PENDING"]
        critical_count = sum(1 for v in active if v.severity == Severity.CRITICAL)
        warning_count  = sum(1 for v in active if v.severity == Severity.WARNING)
        info_count     = len(active) - critical_count - warning_count

        top_risk = ""
        if violations:
            top_violation = max(violations, key=lambda v: v.unaddressed_risk_score)
            top_risk = top_violation.plain_english or top_violation.description

        # Gemini badge text for the UI
        mode_tag = "Hybrid GNN+XGBoost" if gnn_engine.inference_mode == "hybrid" else "GNN"
        if gemini_enriched:
            enriched_count = sum(
                1 for v in violations if v.plain_english and v.plain_english != v.description
            )
            gemini_badge = (
                f"Gemini AI: enriched {enriched_count} finding(s)"
                + (f" | {len(gemini_good_items)} strength(s) identified" if gemini_good_items else "")
                + f" | {mode_tag}"
            )
        else:
            gemini_badge = f"Gemini AI: rules-only mode | {mode_tag}"

        yml_path = _save_validation_yaml(part)

        result = ValidationResult(
            part_id=part.filename,
            overall_manufacturability_score=score,
            risk_summary={
                "critical_count": critical_count,
                "warning_count":  warning_count,
                "info_count":     info_count,
                "top_risk":       top_risk,
                "gemini_badge":   gemini_badge,
                "yml_path":       yml_path,
            },
            violations=violations,
            passed_checks=passed_checks_friendly,
            engineer_review_required=critical_count > 0 or (gnn_score > 0.8),
            confidence=0.95 if len(violations) > 0 or gnn_score > 0 else 0.7,
            gnn_risk_score=round(gnn_score, 4),
            gnn_anomaly=gnn_anomaly_data,
            gemini_enriched=gemini_enriched,
            process=part.process,
            material=part.material,
            face_health=face_health,
        )
        try:
            result.gemini_narrative = await call_gemini_with_timeout(
                generate_gemini_report_narrative,
                result,
                timeout_seconds=8.0,
                fallback=""
            )
        except Exception as narr_ex:
            logger.error(f"Failed to pre-generate narrative: {narr_ex}")
            result.gemini_narrative = ""
        return result

    except Exception as ex:
        logger.error("Unexpected error in validation orchestrator:")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "An unexpected error occurred during validation. Check the server logs.",
                "code": "INTERNAL_ERROR",
            },
        )
