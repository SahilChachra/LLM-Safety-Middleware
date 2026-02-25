# LLM Safety Middleware — Project Overview

## What is this?

A **production-grade safety middleware** that acts as a proxy between clients
and a remote LLM backend (Ollama, OpenAI-compatible API, or any custom
endpoint).  Every request is safety-checked before being forwarded; every
response is safety-checked before being returned.

```
Client → [Safety Middleware] → Remote LLM (Ollama / OpenAI / custom)
                ↕
     Multi-layer safety pipeline
```

---

## Key Features

### Multi-Layer Safety Architecture
1. **Rate Limiter** — Sliding-window per-client throttling
2. **Token-Level Filtering** — Regex-based harmful token detection
3. **Rule Engine** — spaCy NLP: instruction patterns, verb-object combos, entity checks
4. **Semantic Classifier** — Transformer-based (toxic-bert) safety scoring
5. **Post-Generation Validation** — Output content re-checked before returning to client

### Production-Ready
- ✅ Async HTTP proxy with exponential-backoff retry
- ✅ Fail-closed: errors in classifier → CRITICAL rejection
- ✅ Comprehensive error handling and structured logging
- ✅ Configurable safety thresholds per environment
- ✅ Detailed safety reports and analytics
- ✅ Thread-safe statistics with deep-copy protection
- ✅ Context manager support (`with SafetyPipeline(...) as p:`)
- ✅ RESTful API (FastAPI) with admin auth
- ✅ 74-test suite (unit + async integration)

---

## Project Structure

```
llm-safety-pipeline/
├── llm_safety_pipeline.py      # Core library — all classes and pipeline logic
├── api_server.py               # FastAPI HTTP server (middleware entry point)
├── demo_safety_pipeline.py     # 8 demo examples
├── test_safety_pipeline.py     # pytest test suite (74 tests)
│
├── requirements.txt            # Python dependencies
├── config_examples.txt         # Sample JSON configs for each environment
│
├── README.md                   # Full documentation
├── QUICKSTART.md               # 5-minute quick start guide
├── DEPLOYMENT.md               # Production deployment guide
└── PROJECT_OVERVIEW.md         # This file
```

---

## Quick Start

### 1. Install

```bash
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Input-only safety check (sync)

```python
from llm_safety_pipeline import SafetyPipeline

pipeline = SafetyPipeline()
status, report = pipeline.process("What is AI?")
print(f"Status: {status}, Level: {report.safety_level.name}")
```

### 3. Full middleware pipeline (async — requires remote LLM)

```python
import asyncio
from llm_safety_pipeline import LLMBackendConfig, SafetyPipeline

pipeline = SafetyPipeline(
    backend_config=LLMBackendConfig(
        backend_type="ollama",
        base_url="http://localhost:11434",
        model="llama2",
    )
)

async def main():
    status, report = await pipeline.async_process("Explain AI safety.")
    print(report.generated_text)

asyncio.run(main())
```

### 4. Start the API server

```bash
LLM_BACKEND_TYPE=ollama LLM_BASE_URL=http://localhost:11434 \
  LLM_MODEL=llama2 python api_server.py
# Interactive docs: http://localhost:8000/docs
```

---

## Core Classes (`llm_safety_pipeline.py`)

### `SafetyConfig`
Safety-layer configuration: thresholds, which layers are enabled, logging.
*Does NOT include LLM model/generation settings — those go in `LLMBackendConfig`.*

```python
SafetyConfig(
    safety_model_name="unitary/toxic-bert",
    safety_threshold=0.75,
    toxicity_threshold=0.70,
    enable_pattern_matching=True,
    enable_token_filtering=True,
    enable_semantic_check=True,
    enable_post_generation_check=True,
    enable_rate_limiting=True,
    max_requests_per_minute=60,
    max_prompt_length=10_000,
    log_level="INFO",
    save_reports=True,
    custom_banned_patterns=[],
    custom_allowed_contexts=[],
)
```

### `LLMBackendConfig`
Connection and generation settings for the remote LLM.

```python
LLMBackendConfig(
    backend_type="ollama",   # "ollama" | "openai" | "custom"
    base_url="http://localhost:11434",
    model="llama2",
    api_key=None,            # Bearer token for OpenAI etc.
    timeout_seconds=60.0,
    max_retries=3,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    system_prompt=None,
)
# Can also be constructed from env vars:
backend = LLMBackendConfig.from_env()
```

### `ExternalLLMClient`
Async HTTP client that handles Ollama, OpenAI-compatible, and custom backends.
Implements exponential-backoff retry (skips retry on 4xx).

### `SafetyPipeline`
Main orchestrator.

| Method | Description |
|---|---|
| `process(prompt, client_id)` | Sync: runs input safety checks only. No LLM call. |
| `async_process(prompt, client_id, generation_kwargs)` | Async: full pipeline — input checks → LLM → output checks. |
| `backend_health()` | Async: probes the remote LLM reachability. |
| `get_statistics()` | Returns deepcopy of accumulated stats. |
| `reset_statistics()` | Clears stats counters. |
| `add_custom_filter(name, fn)` | Register a `(str) → bool` callable as an extra layer. |
| `async_close()` | Closes the underlying `httpx.AsyncClient`. |

### `SafetyReport`
Per-request record returned by `process()` / `async_process()`.

| Field | Type | Description |
|---|---|---|
| `status` | `str` | `"ACCEPTED"` or `"REJECTED"` |
| `safety_level` | `SafetyLevel` | SAFE / LOW_RISK / MEDIUM_RISK / HIGH_RISK / CRITICAL |
| `rejection_reason` | `Optional[RejectionReason]` | Why it was rejected (if at all) |
| `safety_score` | `Optional[float]` | Semantic classifier score |
| `matched_patterns` | `List[str]` | Pattern names that matched |
| `flagged_tokens` | `List[str]` | Tokens flagged by the token filter |
| `generated_text` | `Optional[str]` | LLM response (if generated) |
| `generation_time` | `Optional[float]` | Seconds spent waiting for LLM |
| `token_filter_passed` | `Optional[bool]` | `None` = not run |
| `pattern_check_passed` | `Optional[bool]` | `None` = not run |
| `semantic_check_passed` | `Optional[bool]` | `None` = not run |
| `post_gen_check_passed` | `Optional[bool]` | `None` = not run |

### Enums

**`SafetyLevel`**: `SAFE` / `LOW_RISK` / `MEDIUM_RISK` / `HIGH_RISK` / `CRITICAL`

**`RejectionReason`**: `TOKEN_FILTER` / `PATTERN_MATCH` / `SEMANTIC_UNSAFE` /
`POST_GEN_UNSAFE` / `RATE_LIMIT` / `MODEL_ERROR` / `CUSTOM_FILTER` /
`INPUT_TOO_LONG` / `BACKEND_UNAVAILABLE`

### Convenience helpers

```python
create_pipeline(config_path=None)     # Creates SafetyPipeline from optional JSON
quick_check(prompt, config=None)      # Returns {is_safe, safety_level, safety_score, reasons}
```

---

## API Server (`api_server.py`)

FastAPI application that wraps the pipeline.

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/v1/check` | POST | — | Input safety check (no LLM) |
| `/api/v1/generate` | POST | — | Full pipeline: check → LLM → check |
| `/api/v1/statistics` | GET | — | Aggregated stats |
| `/api/v1/statistics/reset` | POST | `X-API-Key` | Reset stats |
| `/api/v1/config` | GET | — | Sanitised safety config |
| `/health` | GET | — | Server health + uptime |
| `/health/backend` | GET | — | Remote LLM reachability probe |
| `/dashboard` | GET | — | Built-in monitoring dashboard UI |

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `CONFIG_PATH` | `config_production.json` | Safety JSON config path |
| `ADMIN_API_KEY` | — | Required to use `/statistics/reset` |
| `ALLOWED_ORIGINS` | (none) | Comma-separated CORS origins |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `WORKERS` | `1` | uvicorn workers |
| `LLM_*` | (see above) | LLM backend settings |

---

## Pipeline Flow

```
Incoming request
       │
       ▼
Rate Limiter ─────────────────► RATE_LIMIT rejection
       │
       ▼
Input length guard ───────────► INPUT_TOO_LONG rejection
       │
       ▼
Token-Level Filter ───────────► TOKEN_FILTER rejection
       │
       ▼
Rule Engine (spaCy NLP) ──────► PATTERN_MATCH rejection
       │
       ▼
Semantic Classifier (toxic-bert) ► SEMANTIC_UNSAFE rejection
       │
       ▼  (input is safe — forward to LLM)
External LLM (async HTTP) ────► BACKEND_UNAVAILABLE / MODEL_ERROR
       │
       ▼
Post-Generation Check ────────► POST_GEN_UNSAFE rejection
       │
       ▼
ACCEPTED → return generated text
```

---

## Testing

```bash
# Run all 74 tests
pytest test_safety_pipeline.py -v

# With coverage
pytest --cov=llm_safety_pipeline --cov-report=html
```

Test classes:
- `TestSafetyConfig`, `TestLLMBackendConfig`
- `TestPatternRegistry`, `TestRateLimiter`
- `TestTokenLevelFilter`, `TestRuleEngineFilter`
- `TestSemanticSafetyClassifier` (mocked — no model download needed in CI)
- `TestExternalLLMClient` (respx HTTP mocking)
- `TestSafetyPipeline` — sync `process()`
- `TestSafetyPipelineAsync` — async `async_process()`
- `TestHelpers`, `TestEdgeCases`

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch`, `transformers` | Semantic safety classifier (toxic-bert) |
| `spacy` + `en_core_web_sm` | NLP rule engine |
| `httpx` | Async HTTP client for remote LLM calls |
| `fastapi`, `uvicorn` | REST API server |
| `pydantic` | Request/response models |
| `pytest-asyncio`, `respx` | Async test support + httpx mocking |

---

**Version**: 2.0.0 — Middleware architecture
**Last Updated**: 2026-02-24
