"""
FastAPI Middle-Layer Safety Server
====================================

Sits between clients and a remote LLM backend.  Every request is safety-checked
before being forwarded; every response is safety-checked before being returned.

Environment variables
---------------------
Safety / server
    ADMIN_API_KEY       Secret for admin endpoints (required in production)
    ALLOWED_ORIGINS     Comma-separated CORS origins (default: none)
    HOST                Bind host (default: 0.0.0.0)
    PORT                Bind port (default: 8000)
    WORKERS             uvicorn worker count (default: 1)
    CONFIG_PATH         Path to safety JSON config (default: config_production.json)

LLM backend (all forwarded to LLMBackendConfig.from_env())
    LLM_BACKEND_TYPE    ollama | openai | custom  (default: ollama)
    LLM_BASE_URL        Remote LLM server URL     (default: http://localhost:11434)
    LLM_MODEL           Model name                (default: llama2)
    LLM_API_KEY         Bearer token (OpenAI etc.)
    LLM_TIMEOUT_SECONDS Request timeout           (default: 60)
    LLM_MAX_RETRIES     Retry attempts            (default: 3)
    LLM_MAX_NEW_TOKENS  Default token limit       (default: 512)
    LLM_TEMPERATURE     Default temperature       (default: 0.7)
    LLM_TOP_P           Default top-p             (default: 0.95)
    LLM_SYSTEM_PROMPT   System message (OpenAI)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from llm_safety_pipeline import (
    LLMBackendConfig,
    SafetyConfig,
    SafetyPipeline,
)

# ---------------------------------------------------------------------------
# Server-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------
_ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_admin(key: Optional[str] = Security(_api_key_header)) -> None:
    if not _ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin operations are disabled (ADMIN_API_KEY not configured).",
        )
    if key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key.")


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_pipeline: Optional[SafetyPipeline] = None
_start_time: float = time.time()


def _get_pipeline() -> SafetyPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized.")
    return _pipeline


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _start_time

    # Safety config
    cfg_path = os.environ.get("CONFIG_PATH", "config_production.json")
    try:
        safety_cfg = SafetyConfig.from_json(cfg_path)
        logger.info(f"Loaded safety config from {cfg_path!r}.")
    except FileNotFoundError:
        logger.warning(f"{cfg_path!r} not found — using default safety config.")
        safety_cfg = SafetyConfig()

    # LLM backend config (fully from env vars)
    backend_cfg = LLMBackendConfig.from_env()
    logger.info(
        f"LLM backend: {backend_cfg.backend_type} @ {backend_cfg.base_url} "
        f"model={backend_cfg.model!r}"
    )

    _pipeline = SafetyPipeline(config=safety_cfg, backend_config=backend_cfg)
    _start_time = time.time()
    logger.info("Safety middleware server started.")

    yield  # ── application runs ──

    if _pipeline:
        await _pipeline.async_close()
    logger.info("Safety middleware server shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Safety Middleware",
    description=(
        "Middle-layer safety proxy: validates prompts, forwards safe ones "
        "to the configured LLM backend, validates responses before returning them."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — opt-in via ALLOWED_ORIGINS env var; default is no cross-origin access.
_cors_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _timing_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - t0:.4f}"
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SafetyCheckRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10_000)
    client_id: Optional[str] = Field("default", min_length=1, max_length=128)

    model_config = {
        "json_schema_extra": {
            "example": {"prompt": "What is machine learning?", "client_id": "user_42"}
        }
    }


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10_000)
    client_id: Optional[str] = Field("default", min_length=1, max_length=128)
    # Per-request LLM generation overrides (optional)
    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Explain transformer models in one paragraph.",
                "client_id": "user_42",
                "max_new_tokens": 200,
                "temperature": 0.7,
            }
        }
    }


class SafetyCheckResponse(BaseModel):
    status: str
    safety_level: str
    safety_score: Optional[float]
    rejection_reason: Optional[str]
    matched_patterns: list
    processing_time: float
    timestamp: str


class GenerationResponse(BaseModel):
    status: str
    generated_text: Optional[str]
    safety_level: str
    rejection_reason: Optional[str]
    generation_time: Optional[float]
    processing_time: float
    timestamp: str


class StatisticsResponse(BaseModel):
    total_requests: int
    accepted: int
    rejected: int
    rejection_reasons: Dict[str, int]
    avg_processing_time: float
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    timestamp: str


class BackendHealthResponse(BaseModel):
    reachable: bool
    backend_type: str
    base_url: str
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "LLM Safety Middleware",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "check":      "POST /api/v1/check",
            "generate":   "POST /api/v1/generate",
            "statistics": "GET  /api/v1/statistics",
            "health":     "GET  /health",
            "backend":    "GET  /health/backend",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy" if _pipeline is not None else "initializing",
        version="2.0.0",
        uptime_seconds=time.time() - _start_time,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health/backend", response_model=BackendHealthResponse, tags=["Health"])
async def backend_health():
    """Probe the configured remote LLM backend."""
    p = _get_pipeline()
    try:
        result = await p.backend_health()
    except Exception as exc:
        logger.error(f"backend_health error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backend health check failed.")
    return BackendHealthResponse(**result)


@app.post("/api/v1/check", response_model=SafetyCheckResponse, tags=["Safety"])
async def check_safety(req: SafetyCheckRequest):
    """
    Validate a prompt through all input safety layers without calling the LLM.

    Use this to pre-screen prompts or for offline safety auditing.
    """
    p = _get_pipeline()
    t0 = time.time()
    try:
        status, report = p.process(
            prompt=req.prompt,
            client_id=req.client_id or "default",
        )
    except Exception as exc:
        logger.error(f"check_safety error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Safety check failed.")

    return SafetyCheckResponse(
        status=status,
        safety_level=report.safety_level.name,
        safety_score=report.safety_score,
        rejection_reason=(
            report.rejection_reason.value if report.rejection_reason else None
        ),
        matched_patterns=report.matched_patterns,
        processing_time=time.time() - t0,
        timestamp=report.timestamp,
    )


@app.post("/api/v1/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate(req: GenerationRequest):
    """
    Full middleware pipeline:
    1. Validate input safety
    2. Forward to remote LLM (async)
    3. Validate LLM output safety
    4. Return result
    """
    p = _get_pipeline()
    t0 = time.time()

    # Collect only the overrides that were explicitly provided.
    gen_kwargs: Dict[str, Any] = {}
    if req.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p

    try:
        status, report = await p.async_process(
            prompt=req.prompt,
            client_id=req.client_id or "default",
            generation_kwargs=gen_kwargs or None,
        )
    except Exception as exc:
        logger.error(f"generate error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed.")

    return GenerationResponse(
        status=status,
        generated_text=report.generated_text,
        safety_level=report.safety_level.name,
        rejection_reason=(
            report.rejection_reason.value if report.rejection_reason else None
        ),
        generation_time=report.generation_time,
        processing_time=time.time() - t0,
        timestamp=report.timestamp,
    )


@app.get("/api/v1/statistics", response_model=StatisticsResponse, tags=["Monitoring"])
async def get_statistics():
    """Aggregated request / safety statistics."""
    p = _get_pipeline()
    try:
        stats = p.get_statistics()
    except Exception as exc:
        logger.error(f"get_statistics error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics.")

    return StatisticsResponse(
        total_requests=stats["total_requests"],
        accepted=stats["accepted"],
        rejected=stats["rejected"],
        rejection_reasons=dict(stats["rejection_reasons"]),
        avg_processing_time=stats["avg_processing_time"],
        uptime_seconds=time.time() - _start_time,
    )


@app.post(
    "/api/v1/statistics/reset",
    tags=["Monitoring"],
    dependencies=[Depends(_require_admin)],
)
async def reset_statistics():
    """
    Reset all statistics.  Requires ``X-API-Key: <ADMIN_API_KEY>`` header.
    """
    p = _get_pipeline()
    try:
        p.reset_statistics()
    except Exception as exc:
        logger.error(f"reset_statistics error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset statistics.")
    return {"message": "Statistics reset."}


@app.get("/api/v1/config", tags=["Configuration"])
async def get_config():
    """Return non-sensitive safety configuration fields."""
    p = _get_pipeline()
    cfg = p.config.to_dict()
    return {
        "safety_threshold":          cfg["safety_threshold"],
        "toxicity_threshold":        cfg["toxicity_threshold"],
        "max_prompt_length":         cfg["max_prompt_length"],
        "enable_pattern_matching":   cfg["enable_pattern_matching"],
        "enable_token_filtering":    cfg["enable_token_filtering"],
        "enable_semantic_check":     cfg["enable_semantic_check"],
        "enable_post_generation_check": cfg["enable_post_generation_check"],
        "enable_rate_limiting":      cfg["enable_rate_limiting"],
        "max_requests_per_minute":   cfg["max_requests_per_minute"],
        "backend_type":              p.backend_config.backend_type,
        "backend_model":             p.backend_config.model,
    }


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error.",
            "timestamp": datetime.now().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        workers=int(os.environ.get("WORKERS", "1")),
        log_level="info",
    )
