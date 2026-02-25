# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

## [1.0.0] — 2024-01-01

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
