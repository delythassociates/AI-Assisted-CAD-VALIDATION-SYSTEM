import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root so `uvicorn backend.main:app` picks up GEMINI_API_KEY
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from contextlib import asynccontextmanager
from .core.config import settings
from .core.database import init_db
from .api.router_validate import router as validate_router
from .api.router_fix import router as fix_router
from .api.router_report import router as report_router
from .api.router_web import router as web_router
from .api.router_flywheel import router as flywheel_router
from .rules.engine import engine
from .rules import injection, cnc, die_casting, assembly, gdt
from .services import gnn_engine

DEV_MODE = os.environ.get("EUREKA_DEV_MODE", "").lower() in ("true", "1", "yes")
API_KEY  = os.environ.get("EUREKA_API_KEY", "eureka-dev-key-change-me")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Seed the model registry with the currently-loaded model so that the
    # /flywheel endpoint always returns real baseline numbers from the first run.
    try:
        from .ml.feedback_store import seed_initial_registry
        seed_initial_registry(
            version  = gnn_engine.model_version,
            auc_roc  = 0.0,   # will be overwritten by first fine-tune run
            f1_score = 0.0,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not seed registry: %s", e)

    yield


app = FastAPI(
    title="HERMES DFM Validation API",
    version="1.0.0",
    lifespan=lifespan,
    debug=DEV_MODE,
    docs_url="/docs" if DEV_MODE else None,
    redoc_url="/redoc" if DEV_MODE else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "null"] if not DEV_MODE else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXEMPT_PATHS = {"/health", "/processes", "/web/presets", "/flywheel"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path not in EXEMPT_PATHS:
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
    return await call_next(request)


app.include_router(validate_router,  prefix="/validate",       tags=["Validation"])
app.include_router(fix_router,       prefix="/fix-suggestion", tags=["Fixes"])
app.include_router(report_router,    prefix="/report",         tags=["Reports"])
app.include_router(web_router,       prefix="/web",            tags=["web"])
app.include_router(flywheel_router,  prefix="/flywheel",       tags=["Flywheel"])

# Mount /feedback under the flywheel router prefix
app.include_router(flywheel_router,  prefix="/feedback",       tags=["Feedback"])


@app.get("/health")
async def health():
    if gnn_engine.session is not None:
        inference_path = "onnx"
    elif gnn_engine.model is not None:
        inference_path = "pytorch"
    else:
        inference_path = "none"

    # Gemini badge for Task Pane UI
    if gnn_engine.inference_mode == "hybrid":
        gemini_model_badge = "Hybrid GNN+XGBoost"
    else:
        gemini_model_badge = "GNN Neural Network"

    return {
        "status":            "ok",
        "model_loaded":      gnn_engine.model is not None,
        "gnn_available":     gnn_engine.available,
        "onnx_loaded":       gnn_engine.session is not None,
        "inference_path":    inference_path,
        "inference_mode":    gnn_engine.inference_mode,
        "model_version":     gnn_engine.model_version,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "gnn_threshold":     gnn_engine.threshold,
        "gemini_badge":      gemini_model_badge,
    }


@app.get("/processes")
async def list_processes():
    return engine.get_available_processes()
