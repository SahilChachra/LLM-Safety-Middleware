# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Built-in monitoring dashboard served at `GET /dashboard` — auto-refreshes every
  10 seconds; shows request counts, accept rate, rejection reason breakdown,
  avg latency, and backend health status. No external dependencies (pure HTML/CSS/JS).

### Fixed

- `SafetyPipeline.__init__` now defaults to `LLMBackendConfig.from_env()` instead
  of `LLMBackendConfig()` (hardcoded values), so `LLM_BACKEND_TYPE`, `LLM_BASE_URL`,
  `LLM_MODEL`, and related env vars are respected by every pipeline instance — not just
  `api_server.py`.
- `demo_safety_pipeline.py` updated to use `LLMBackendConfig.from_env()` in all
  examples so env vars set at the shell are picked up correctly.
- Ollama health probe now uses the server root URL (`GET /`) instead of `GET /api/tags`.
  The root endpoint is the canonical Ollama liveness check and is less likely to be
  restricted or slow on remote servers.
- Health probe failure log level lowered from `WARNING` to `DEBUG` to avoid log spam
  when the probe fails but actual generation requests succeed.
- `config_production.json` — `reports_dir` changed from `/app/safety_reports`
  (Docker-only absolute path) to `./safety_reports` (relative path that resolves
  correctly both locally and inside the container).
- GitHub Actions `docker.yml` — added `load: true` to `docker/build-push-action` so
  the built image is loaded into the local Docker daemon and available to the
  subsequent `docker run` smoke-test step.
- Test mocks for `test_health_check_reachable` and `test_health_check_unreachable`
  updated to match the new Ollama probe URL (`/` instead of `/api/tags`).

---

## [2.0.0] — 2026-02-24

### Breaking Changes

- **Architecture change**: local GPT-2 generator completely removed; the pipeline
  is now a pure **async HTTP proxy** to a remote LLM backend (Ollama,
  OpenAI-compatible, or custom).
- `SafetyConfig` — removed fields: `llm_model_name`, `max_new_tokens`,
  `temperature`, `top_p`, `batch_size`, `max_cache_size`.  These are now
  configured via `LLMBackendConfig` / environment variables.
- `SafetyPipeline.process()` — `generate=` keyword argument removed.
  `process()` is now input-only (sync, no LLM call).

### Added

- `LLMBackendConfig` dataclass with `from_env()` class method — supports
  `ollama`, `openai`, and `custom` backend types.
- `ExternalLLMClient` — async `httpx`-based HTTP client with exponential-backoff
  retry (does not retry on 4xx errors).
- `SafetyPipeline.async_process()` — full async pipeline:
  input safety checks → remote LLM → output safety checks.
- `SafetyPipeline.async_close()` — graceful shutdown of the underlying
  `httpx.AsyncClient`.
- `SafetyPipeline.backend_health()` — async probe of the remote LLM endpoint.
- `RejectionReason.BACKEND_UNAVAILABLE` and `RejectionReason.INPUT_TOO_LONG`.
- `SafetyConfig.max_prompt_length` (default 10 000 characters) — input length guard.
- `GET /health/backend` API endpoint — probes the remote LLM reachability.
- `POST /api/v1/statistics/reset` now requires `X-API-Key: <ADMIN_API_KEY>` header.
- Full async test suite: 74 tests using `pytest-asyncio` and `respx` for httpx mocking.
- `pyproject.toml`, `LICENSE`, `.gitignore`, `.env.example`, `CONTRIBUTING.md`,
  `CHANGELOG.md`, GitHub Actions CI workflows.

### Changed

- CPU-bound PyTorch/spaCy inference offloaded to a thread-pool executor so the
  asyncio event loop stays unblocked.
- `SafetyReport` check-passed flags changed to `Optional[bool]` — `None` means
  the check was not run (disabled or not reached), preventing false "passed" signals.
- `get_statistics()` now returns a `copy.deepcopy()` to prevent callers from
  mutating live counters.
- `device` in `SafetyConfig` now uses `field(default_factory=...)` to detect
  CUDA safely at construction time.
- Safe-context regex no longer bypasses all downstream layers — it now defers to
  the semantic classifier for a final verdict.
- `SemanticSafetyClassifier` correctly handles `unitary/toxic-bert`'s multi-label
  output (sigmoid + `probs.max()`) instead of binary softmax.
- CORS: `allow_credentials=False`; origins opt-in via `ALLOWED_ORIGINS` env var.
- FastAPI `@app.on_event` replaced with `@asynccontextmanager lifespan`.

### Fixed

- `_has_prohibited_entities` now checks both NER labels and plain text, fixing
  false-negatives for entities spaCy did not label.
- `_has_harmful_instruction_pattern` uses dependency-tree traversal instead of
  a flat token scan, reducing false-positives.
- `add_custom_filter` now properly raises on duplicate filter names.
- `subprocess.run` in model download helper uses `sys.executable` so the correct
  Python is invoked inside virtual environments.

---

## [1.0.0] — 2025-02-01

### Added

- Initial release: multi-layer LLM safety pipeline with local GPT-2 generation.
- `TokenLevelFilter`, `RuleEngineFilter`, `SemanticSafetyClassifier`, `SafeLLMGenerator`.
- `SafetyPipeline.process(prompt, generate=True/False)`.
- FastAPI server with `/check`, `/generate`, `/statistics` endpoints.
- Docker + docker-compose support.

---

[Unreleased]: https://github.com/SahilChachra/LLM-Safety-Middleware/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/SahilChachra/LLM-Safety-Middleware/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/SahilChachra/LLM-Safety-Middleware/releases/tag/v1.0.0
