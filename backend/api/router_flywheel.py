"""
router_flywheel.py — Active Learning Flywheel API endpoints.

Endpoints:
  POST /feedback          — Engineer submits a label correction
  GET  /feedback/stats    — Aggregate correction statistics
  GET  /flywheel          — Full flywheel status (100% real data)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FeedbackPayload(BaseModel):
    part_id: str
    yml_path: Optional[str] = None
    predicted_label: int          # 0 = clean, 1 = defective
    predicted_score: float        # raw probability from inference engine
    engineer_label: int           # 0 = clean, 1 = defective


# ---------------------------------------------------------------------------
# Lazy imports of ML layer (avoids circular import at startup)
# ---------------------------------------------------------------------------

def _get_feedback_store():
    from ..ml.feedback_store import (
        log_feedback,
        log_inference,
        get_stats,
        get_uncollected_corrections,
        get_current_registry_entry,
        get_all_registry_entries,
        get_total_inference_count,
        seed_initial_registry,
    )
    return (log_feedback, log_inference, get_stats, get_uncollected_corrections,
            get_current_registry_entry, get_all_registry_entries,
            get_total_inference_count, seed_initial_registry)


def _get_engine():
    from ..services import gnn_engine
    return gnn_engine


def _get_trigger():
    from ..ml.fine_tune_trigger import check_and_trigger
    return check_and_trigger


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

@router.post("")
async def submit_feedback(
    payload: FeedbackPayload,
    background_tasks: BackgroundTasks,
):
    """
    Record an engineer label correction.

    The model_version is read from the *live* inference engine at the time
    the request arrives — always 100% accurate.
    After recording, the fine-tune threshold is checked asynchronously.
    """
    (log_feedback, _, get_stats, get_uncollected_corrections,
     _, _, _, _) = _get_feedback_store()
    engine = _get_engine()

    row_id = log_feedback(
        part_id        = payload.part_id,
        yml_path       = payload.yml_path,
        predicted_label= payload.predicted_label,
        predicted_score= payload.predicted_score,
        engineer_label = payload.engineer_label,
        model_version  = engine.model_version,
    )

    # Schedule threshold check — runs after response is sent
    check_and_trigger = _get_trigger()
    background_tasks.add_task(check_and_trigger)

    stats = get_stats()
    corrections_pending = len(get_uncollected_corrections())

    return {
        "status": "recorded",
        "feedback_id": row_id,
        "is_correction": payload.predicted_label != payload.engineer_label,
        "stats": {
            **stats,
            "corrections_pending": corrections_pending,
            "fine_tune_threshold": 30,
            "corrections_until_fine_tune": max(0, 30 - corrections_pending),
        },
    }


# ---------------------------------------------------------------------------
# GET /feedback/stats
# ---------------------------------------------------------------------------

@router.get("/stats")
async def feedback_stats():
    """Live aggregate statistics — every number from SQLite."""
    (_, _, get_stats, get_uncollected_corrections,
     get_current_registry_entry, _, _, _) = _get_feedback_store()
    engine = _get_engine()

    stats = get_stats()
    registry = get_current_registry_entry()
    corrections_pending = len(get_uncollected_corrections())

    return {
        **stats,
        "current_model_version":  engine.model_version,
        "current_model_trained_on": registry.get("trained_on_n", 0),
        "current_model_auc":      registry.get("auc_roc", 0.0),
        "fine_tune_threshold":    30,
        "corrections_pending":    corrections_pending,
        "corrections_until_fine_tune": max(0, 30 - corrections_pending),
    }


# ---------------------------------------------------------------------------
# GET /flywheel  — full status, 100% real data
# ---------------------------------------------------------------------------

@router.get("")
async def flywheel_status():
    """
    Complete flywheel dashboard payload.

    Every field comes from:
      • SQLite feedback.db  (feedback counts, correction breakdown)
      • model_registry table (version history, AUC, F1, n_graphs)
      • inference engine    (live version string)
      • inference_log table (total parts ever analysed)
    """
    (_, _, get_stats, get_uncollected_corrections,
     get_current_registry_entry, get_all_registry_entries,
     get_total_inference_count, _) = _get_feedback_store()
    engine = _get_engine()

    stats        = get_stats()
    registry     = get_all_registry_entries()
    corrections  = get_uncollected_corrections()
    total_parts  = get_total_inference_count()
    current      = get_current_registry_entry()

    return {
        "current_version": engine.model_version,
        "total_parts_analyzed": total_parts,

        "feedback": {
            "total_collected":           stats["total_feedback"],
            "total_corrections":         stats["total_corrections"],
            "false_positives_corrected": stats["false_positives"],
            "false_negatives_corrected": stats["false_negatives"],
            "corrections_this_week":     stats["corrections_this_week"],
        },

        "fine_tune": {
            "threshold":           30,
            "pending_corrections": len(corrections),
            "next_fine_tune_in":   max(0, 30 - len(corrections)),
            "last_fine_tune_auc":  current.get("auc_roc"),
            "last_fine_tune_f1":   current.get("f1_score"),
        },

        "model_history": [
            {
                "version":      r["version"],
                "auc_roc":      r["auc_roc"],
                "f1_score":     r["f1_score"],
                "trained_on_n": r["trained_on_n"],
                "deployed":     bool(r["deployed"]),
                "deployed_at":  r["deployed_at"],
            }
            for r in registry
        ],
    }
