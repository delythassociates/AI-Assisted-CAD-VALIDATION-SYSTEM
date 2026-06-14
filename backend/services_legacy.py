"""Shared service instances for the application."""
from .ml.inference import DFMInferenceEngine
from .core.config import settings

gnn_engine = DFMInferenceEngine(
    model_path=settings.GNN_MODEL_PATH,
    threshold_path=settings.GNN_THRESHOLD_PATH,
)
