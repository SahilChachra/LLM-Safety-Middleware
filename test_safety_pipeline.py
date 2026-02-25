"""
Test Suite — LLM Safety Middleware Pipeline
============================================

Covers:
  - Configuration (SafetyConfig, LLMBackendConfig)
  - PatternRegistry
  - RateLimiter
  - TokenLevelFilter
  - RuleEngineFilter
  - SemanticSafetyClassifier (mocked)
  - ExternalLLMClient (httpx mocked via respx)
  - SafetyPipeline.process() — sync input-only checks
  - SafetyPipeline.async_process() — full async pipeline (LLM mocked)
  - SafetyReport
  - Convenience helpers
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import respx
import httpx

from llm_safety_pipeline import (
    ExternalLLMClient,
    LLMBackendConfig,
    PatternRegistry,
    RateLimiter,
    RejectionReason,
    SafetyConfig,
    SafetyLevel,
    SafetyPipeline,
    SafetyReport,
    SemanticSafetyClassifier,
    TokenLevelFilter,
    RuleEngineFilter,
    create_pipeline,
    quick_check,
    _pipeline_cache,
)

# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def safety_cfg():
    return SafetyConfig(
        save_reports=False,
        log_level="ERROR",
        enable_semantic_check=False,   # Keep unit tests fast; semantic tested separately.
    )


@pytest.fixture
def backend_cfg():
    return LLMBackendConfig(
        backend_type="ollama",
        base_url="http://fake-llm:11434",
        model="test-model",
        max_retries=1,
    )


@pytest.fixture
def mock_logger():
    import logging
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def token_filter(mock_logger):
    return TokenLevelFilter(patterns=PatternRegistry(), logger=mock_logger)


# ═══════════════════════════════════════════════════════════════════════════
# SafetyConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyConfig:
    def test_defaults(self):
        cfg = SafetyConfig()
        assert cfg.safety_threshold == 0.75
        assert cfg.enable_pattern_matching is True
        assert cfg.max_prompt_length == 10_000
        assert cfg.device in ("cpu", "cuda")

    def test_device_is_evaluated_at_instantiation(self):
        # device must NOT be a class-level constant — each instance computes it.
        cfg1 = SafetyConfig()
        cfg2 = SafetyConfig()
        assert cfg1.device == cfg2.device  # both resolve the same way

    def test_from_dict_ignores_unknown_keys(self):
        d = {"safety_threshold": 0.9, "unknown_future_field": "ignored"}
        cfg = SafetyConfig.from_dict(d)
        assert cfg.safety_threshold == 0.9
        assert not hasattr(cfg, "unknown_future_field")

    def test_to_dict_roundtrip(self, tmp_path):
        cfg = SafetyConfig(safety_threshold=0.88, save_reports=False)
        path = str(tmp_path / "cfg.json")
        cfg.save_json(path)
        loaded = SafetyConfig.from_json(path)
        assert loaded.safety_threshold == 0.88
        assert loaded.save_reports is False


# ═══════════════════════════════════════════════════════════════════════════
# LLMBackendConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMBackendConfig:
    def test_defaults(self):
        cfg = LLMBackendConfig()
        assert cfg.backend_type == "ollama"
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.model == "llama2"

    def test_from_dict_ignores_unknown(self):
        cfg = LLMBackendConfig.from_dict({"model": "gpt4", "unknown": "x"})
        assert cfg.model == "gpt4"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND_TYPE", "openai")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        cfg = LLMBackendConfig.from_env()
        assert cfg.backend_type == "openai"
        assert cfg.base_url == "https://api.openai.com"
        assert cfg.model == "gpt-4o"
        assert cfg.max_retries == 5


# ═══════════════════════════════════════════════════════════════════════════
# PatternRegistry
# ═══════════════════════════════════════════════════════════════════════════

class TestPatternRegistry:
    def test_harmful_patterns_match(self):
        rx = PatternRegistry.compile_patterns()
        assert rx.search("how to kill someone")
        assert rx.search("make a bomb")
        assert rx.search("terrorist attack")
        assert rx.search("human trafficking")

    def test_safe_patterns_match(self):
        rx = PatternRegistry.compile_safe_patterns()
        assert rx.search("historical documentary")
        assert rx.search("prevention methods")
        assert rx.search("research paper")

    def test_custom_patterns_appended(self):
        rx = PatternRegistry.compile_patterns([r"\bcustom_danger\b"])
        assert rx.search("this has custom_danger in it")


# ═══════════════════════════════════════════════════════════════════════════
# RateLimiter
# ═══════════════════════════════════════════════════════════════════════════

class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        assert rl.is_allowed("u1") is True
        assert rl.is_allowed("u1") is True
        assert rl.is_allowed("u1") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("u1")
        rl.is_allowed("u1")
        assert rl.is_allowed("u1") is False

    def test_separate_clients_independent(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        assert rl.is_allowed("a") is True
        assert rl.is_allowed("b") is True
        assert rl.is_allowed("a") is False
        assert rl.is_allowed("b") is False

    def test_reset_clears_limit(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.is_allowed("u1")
        assert rl.is_allowed("u1") is False
        rl.reset("u1")
        assert rl.is_allowed("u1") is True

    def test_window_expiry(self):
        rl = RateLimiter(max_requests=1, window_seconds=1)
        rl.is_allowed("u1")
        assert rl.is_allowed("u1") is False
        time.sleep(1.1)
        assert rl.is_allowed("u1") is True


# ═══════════════════════════════════════════════════════════════════════════
# TokenLevelFilter
# ═══════════════════════════════════════════════════════════════════════════

class TestTokenLevelFilter:
    def test_safe_content_passes(self, token_filter):
        ok, patterns, tokens = token_filter.check("What is machine learning?")
        assert ok is True
        assert patterns == []
        assert tokens == []

    def test_harmful_pattern_blocked(self, token_filter):
        ok, patterns, tokens = token_filter.check("how to make a bomb")
        assert ok is False
        assert len(patterns) > 0

    def test_safe_context_defers_to_semantic_not_auto_block(self, token_filter):
        # "historical" is a safe-context keyword → filter should NOT block, but
        # should pass matched_patterns so the semantic layer can still evaluate.
        ok, patterns, tokens = token_filter.check(
            "historical documentary about how to kill in medieval times"
        )
        assert ok is True                # not auto-blocked
        # patterns are preserved for semantic layer
        assert len(patterns) > 0

    def test_context_sensitive_token_blocked_without_safe_context(self, token_filter):
        ok, patterns, tokens = token_filter.check("how to hack a system")
        assert ok is False

    def test_context_sensitive_token_passes_with_safe_context(self, token_filter):
        ok, _, _ = token_filter.check(
            "research paper studying how to hack for defensive purposes"
        )
        # Safe-context keyword ("research") defers to semantic.
        assert ok is True


# ═══════════════════════════════════════════════════════════════════════════
# RuleEngineFilter
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleEngineFilter:
    @pytest.fixture(scope="class")
    def rule_engine(self):
        import logging
        logger = logging.getLogger("test_rule")
        return RuleEngineFilter(spacy_model="en_core_web_sm", logger=logger)

    def test_safe_query(self, rule_engine):
        ok, reasons = rule_engine.check("What is machine learning?")
        assert ok is True
        assert reasons == []

    def test_harmful_instruction(self, rule_engine):
        ok, reasons = rule_engine.check("how to kill someone")
        assert ok is False
        assert len(reasons) > 0

    def test_dangerous_verb_object(self, rule_engine):
        ok, _ = rule_engine.check("attack the civilian")
        assert ok is False

    def test_prohibited_entity_plain_text(self, rule_engine):
        # "terrorist" appears as plain text — the old code only checked NER labels.
        ok, reasons = rule_engine.check("join the terrorist group")
        assert ok is False

    def test_dependency_based_detection(self, rule_engine):
        # Catches variations the old positional check missed.
        ok, _ = rule_engine.check("guide on hacking")
        assert ok is False


# ═══════════════════════════════════════════════════════════════════════════
# SemanticSafetyClassifier (mocked — avoids downloading models in CI)
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticSafetyClassifier:
    def _make_classifier(self, mock_logits, safety_threshold=0.75):
        import torch
        import logging

        # BatchEncoding is a dict subclass with a .to() method — replicate that.
        class _FakeBatchEncoding(dict):
            def to(self, device):
                return self

        fake_inputs = _FakeBatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = fake_inputs

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        with patch("llm_safety_pipeline.AutoTokenizer.from_pretrained",
                   return_value=mock_tokenizer), \
             patch("llm_safety_pipeline.AutoModelForSequenceClassification.from_pretrained",
                   return_value=mock_model):
            return SemanticSafetyClassifier(
                model_name="mock-model",
                device="cpu",
                safety_threshold=safety_threshold,
                toxicity_threshold=0.7,
                logger=logging.getLogger("test"),
            )

    def test_binary_safe_text(self):
        import torch
        # Binary: [safe_logit=5.0, toxic_logit=0.1]  → softmax → safe ≈ 0.993
        clf = self._make_classifier(torch.tensor([[5.0, 0.1]]))
        is_safe, score, level = clf.check("Hello world")
        assert is_safe is True
        assert score > 0.9
        assert level == SafetyLevel.SAFE

    def test_binary_toxic_text(self):
        import torch
        # Binary: [safe_logit=0.1, toxic_logit=5.0] → toxic ≈ 0.993
        clf = self._make_classifier(torch.tensor([[0.1, 5.0]]))
        is_safe, score, level = clf.check("bad content")
        assert is_safe is False
        assert score < 0.1
        assert level == SafetyLevel.CRITICAL

    def test_multilabel_uses_sigmoid_max(self):
        import torch
        # 6 labels (toxic-bert style): all very low except label[3] (threat=0.9)
        logits = torch.tensor([[0.1, 0.05, 0.1, 2.2, 0.05, 0.1]])
        clf = self._make_classifier(logits)
        is_safe, score, level = clf.check("text with a threat")
        assert is_safe is False   # max sigmoid(2.2) ≈ 0.9 → safe_prob ≈ 0.1 < 0.75

    def test_fail_closed_on_error(self):
        import logging
        with patch("llm_safety_pipeline.AutoTokenizer.from_pretrained",
                   side_effect=RuntimeError("network")), \
             pytest.raises(RuntimeError):
            SemanticSafetyClassifier(
                model_name="bad-model", device="cpu",
                safety_threshold=0.75, toxicity_threshold=0.7,
                logger=logging.getLogger("test"),
            )


# ═══════════════════════════════════════════════════════════════════════════
# ExternalLLMClient (httpx mocked with respx)
# ═══════════════════════════════════════════════════════════════════════════

class TestExternalLLMClient:

    # ── Ollama ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    @respx.mock
    async def test_ollama_generate_success(self):
        respx.post("http://fake-llm:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "Hello from Ollama!", "done": True})
        )
        cfg = LLMBackendConfig(
            backend_type="ollama",
            base_url="http://fake-llm:11434",
            model="llama2",
            max_retries=1,
        )
        import logging
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        text, elapsed = await client.generate("Say hello")
        assert text == "Hello from Ollama!"
        assert elapsed >= 0
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_ollama_passes_options(self):
        route = respx.post("http://fake-llm:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok", "done": True})
        )
        cfg = LLMBackendConfig(
            backend_type="ollama", base_url="http://fake-llm:11434",
            model="llama2", max_retries=1,
        )
        import logging
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        await client.generate("prompt", max_new_tokens=64, temperature=0.5)
        body = json.loads(route.calls[0].request.content)
        assert body["options"]["num_predict"] == 64
        assert body["options"]["temperature"] == 0.5
        await client.close()

    # ── OpenAI-compatible ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    @respx.mock
    async def test_openai_generate_success(self):
        respx.post("http://fake-openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "OpenAI response"}}]
            })
        )
        cfg = LLMBackendConfig(
            backend_type="openai", base_url="http://fake-openai",
            model="gpt-4o", max_retries=1,
        )
        import logging
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        text, _ = await client.generate("Hello")
        assert text == "OpenAI response"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_openai_includes_system_prompt(self):
        route = respx.post("http://fake-openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        cfg = LLMBackendConfig(
            backend_type="openai", base_url="http://fake-openai",
            model="gpt-4o", system_prompt="You are helpful.", max_retries=1,
        )
        import logging
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        await client.generate("hi")
        messages = json.loads(route.calls[0].request.content)["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        await client.close()

    # ── Custom backend ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    @respx.mock
    async def test_custom_generate_success(self):
        respx.post("http://custom-llm/v1/run").mock(
            return_value=httpx.Response(200, json={"output": "custom result"})
        )
        cfg = LLMBackendConfig(
            backend_type="custom", base_url="http://custom-llm",
            model="custom-model",
            custom_endpoint="/v1/run",
            custom_prompt_field="input",
            custom_response_field="output",
            max_retries=1,
        )
        import logging
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        text, _ = await client.generate("hello")
        assert text == "custom result"
        await client.close()

    # ── Retry / error handling ──────────────────────────────────────────────

    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_on_5xx_then_succeeds(self):
        import logging
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(200, json={"response": "recovered", "done": True})

        respx.post("http://fake-llm:11434/api/generate").mock(side_effect=side_effect)
        cfg = LLMBackendConfig(
            backend_type="ollama", base_url="http://fake-llm:11434",
            model="llama2", max_retries=3, retry_delay_seconds=0,
        )
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        text, _ = await client.generate("hi")
        assert text == "recovered"
        assert call_count == 2
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_raises_after_all_retries_exhausted(self):
        import logging
        respx.post("http://fake-llm:11434/api/generate").mock(
            return_value=httpx.Response(503, text="down")
        )
        cfg = LLMBackendConfig(
            backend_type="ollama", base_url="http://fake-llm:11434",
            model="llama2", max_retries=2, retry_delay_seconds=0,
        )
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        with pytest.raises(RuntimeError, match="unavailable after"):
            await client.generate("hi")
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_4xx_not_retried(self):
        import logging
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, text="Bad Request")

        respx.post("http://fake-llm:11434/api/generate").mock(side_effect=side_effect)
        cfg = LLMBackendConfig(
            backend_type="ollama", base_url="http://fake-llm:11434",
            model="llama2", max_retries=3, retry_delay_seconds=0,
        )
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        with pytest.raises(RuntimeError, match="HTTP 400"):
            await client.generate("hi")
        assert call_count == 1   # must NOT retry 4xx
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unknown_backend_type_raises(self):
        import logging
        cfg = LLMBackendConfig(
            backend_type="unknown_backend", base_url="http://x",
            model="m", max_retries=1,
        )
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        with pytest.raises(ValueError, match="Unknown backend_type"):
            await client.generate("hi")
        await client.close()

    # ── Health check ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check_reachable(self):
        import logging
        respx.get("http://fake-llm:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        cfg = LLMBackendConfig(backend_type="ollama", base_url="http://fake-llm:11434", model="llama2")
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        result = await client.health_check()
        assert result["reachable"] is True
        assert result["model"] == "llama2"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check_unreachable(self):
        import logging
        respx.get("http://offline:11434/api/tags").mock(
            side_effect=httpx.ConnectError("refused")
        )
        cfg = LLMBackendConfig(backend_type="ollama", base_url="http://offline:11434", model="x")
        client = ExternalLLMClient(cfg, logging.getLogger("test"))
        result = await client.health_check()
        assert result["reachable"] is False
        await client.close()


# ═══════════════════════════════════════════════════════════════════════════
# SafetyReport
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyReport:
    def test_check_flags_default_to_none(self):
        r = SafetyReport(
            timestamp="2024-01-01T00:00:00",
            prompt="test",
            status="ACCEPTED",
            safety_level=SafetyLevel.SAFE,
        )
        assert r.token_filter_passed is None
        assert r.pattern_check_passed is None
        assert r.semantic_check_passed is None
        assert r.post_gen_check_passed is None

    def test_to_dict_serialises_level_name(self):
        r = SafetyReport(
            timestamp="2024-01-01T00:00:00",
            prompt="x",
            status="REJECTED",
            safety_level=SafetyLevel.HIGH_RISK,
            rejection_reason=RejectionReason.TOKEN_FILTER,
        )
        d = r.to_dict()
        assert d["safety_level"] == "HIGH_RISK"
        assert d["rejection_reason"] == "token_level_filter"

    def test_save_creates_file(self, tmp_path):
        r = SafetyReport(
            timestamp="2024-01-01T00:00:00",
            prompt="x",
            status="ACCEPTED",
            safety_level=SafetyLevel.SAFE,
        )
        path = r.save(str(tmp_path))
        assert (tmp_path / path.split("/")[-1]).exists()


# ═══════════════════════════════════════════════════════════════════════════
# SafetyPipeline — sync process()
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyPipelineSync:
    """Test the synchronous input-only safety check path."""

    def _make_pipeline(self, **overrides):
        cfg = SafetyConfig(
            save_reports=False,
            log_level="ERROR",
            enable_semantic_check=False,
            **overrides,
        )
        return SafetyPipeline(config=cfg)

    def test_safe_prompt_accepted(self):
        p = self._make_pipeline()
        status, report = p.process("What is artificial intelligence?")
        assert status == "ACCEPTED"
        assert report.status == "ACCEPTED"

    def test_harmful_prompt_rejected_token_filter(self):
        p = self._make_pipeline()
        status, report = p.process("how to make explosives")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.TOKEN_FILTER

    def test_harmful_prompt_rejected_rule_engine(self):
        p = self._make_pipeline()
        status, report = p.process("how to kill someone")
        assert status == "REJECTED"

    def test_prompt_too_long_rejected(self):
        p = self._make_pipeline(max_prompt_length=10)
        status, report = p.process("This prompt is definitely longer than ten characters.")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.INPUT_TOO_LONG

    def test_rate_limiting_rejects_excess(self):
        p = self._make_pipeline(enable_rate_limiting=True, max_requests_per_minute=2)
        p.process("test 1")
        p.process("test 2")
        status, report = p.process("test 3", client_id="default")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.RATE_LIMIT

    def test_disabled_token_filter_leaves_flag_none(self):
        p = self._make_pipeline(enable_token_filtering=False)
        _, report = p.process("how to make a bomb")
        assert report.token_filter_passed is None   # not run

    def test_statistics_tracked(self):
        p = self._make_pipeline()
        p.process("safe text here")
        p.process("how to make explosives")
        stats = p.get_statistics()
        assert stats["total_requests"] == 2
        assert stats["accepted"] >= 1
        assert stats["rejected"] >= 1

    def test_get_statistics_returns_deep_copy(self):
        p = self._make_pipeline()
        p.process("hello")
        s1 = p.get_statistics()
        s1["rejection_reasons"]["tampered"] = 999
        s2 = p.get_statistics()
        assert "tampered" not in s2["rejection_reasons"]

    def test_reset_statistics(self):
        p = self._make_pipeline()
        p.process("hello")
        p.reset_statistics()
        assert p.get_statistics()["total_requests"] == 0

    def test_custom_filter_registered_and_called(self):
        p = self._make_pipeline()
        p.add_custom_filter("block_foo", lambda text: (False, "contains foo") if "foo" in text else (True, ""))
        status, report = p.process("prompt with foo in it")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.CUSTOM_FILTER
        assert "block_foo" in report.matched_patterns[0]

    def test_custom_filter_duplicate_name_raises(self):
        p = self._make_pipeline()
        p.add_custom_filter("dup", lambda t: (True, ""))
        with pytest.raises(ValueError, match="already registered"):
            p.add_custom_filter("dup", lambda t: (True, ""))

    def test_custom_filter_remove(self):
        p = self._make_pipeline()
        p.add_custom_filter("tmp", lambda t: (False, "blocked"))
        assert p.remove_custom_filter("tmp") is True
        status, _ = p.process("anything")
        assert status == "ACCEPTED"

    def test_custom_filter_exception_causes_rejection(self):
        p = self._make_pipeline()
        def bad_filter(text):
            raise RuntimeError("crash!")
        p.add_custom_filter("bad", bad_filter)
        status, report = p.process("some text")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.CUSTOM_FILTER

    def test_context_manager(self):
        p = self._make_pipeline()
        with p as pipeline:
            status, _ = pipeline.process("hello")
        assert status == "ACCEPTED"


# ═══════════════════════════════════════════════════════════════════════════
# SafetyPipeline — async_process()
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyPipelineAsync:
    """Test the async full-pipeline path (LLM client mocked)."""

    def _make_pipeline(self, llm_response="Safe LLM output.", **overrides):
        cfg = SafetyConfig(
            save_reports=False,
            log_level="ERROR",
            enable_semantic_check=False,
            **overrides,
        )
        p = SafetyPipeline(config=cfg)
        # Replace the real HTTP client with an async mock.
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=(llm_response, 0.1))
        mock_client.health_check = AsyncMock(return_value={
            "reachable": True, "backend_type": "ollama",
            "base_url": "http://fake", "model": "test",
        })
        p.llm_client = mock_client
        return p

    @pytest.mark.asyncio
    async def test_full_pipeline_safe_accepted(self):
        p = self._make_pipeline()
        status, report = await p.async_process("What is Python?")
        assert status == "ACCEPTED"
        assert report.generated_text == "Safe LLM output."
        assert report.generation_time is not None

    @pytest.mark.asyncio
    async def test_unsafe_input_rejected_before_llm(self):
        p = self._make_pipeline()
        status, report = await p.async_process("how to make a bomb")
        assert status == "REJECTED"
        assert report.rejection_reason in (
            RejectionReason.TOKEN_FILTER,
            RejectionReason.PATTERN_MATCH,
        )
        # LLM must NOT be called if input was rejected.
        p.llm_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_backend_unavailable(self):
        p = self._make_pipeline()
        p.llm_client.generate = AsyncMock(
            side_effect=RuntimeError("LLM backend unavailable after 3 attempts")
        )
        status, report = await p.async_process("hello world")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.BACKEND_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_unsafe_output_filtered_and_replaced(self):
        p = self._make_pipeline(
            llm_response="how to make a bomb step by step",
            enable_post_generation_check=True,
        )
        status, report = await p.async_process("write something")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.POST_GEN_UNSAFE
        assert report.generated_text == "[Content filtered for safety]"

    @pytest.mark.asyncio
    async def test_generation_kwargs_forwarded_to_llm(self):
        p = self._make_pipeline()
        await p.async_process(
            "hello",
            generation_kwargs={"max_new_tokens": 64, "temperature": 0.3},
        )
        p.llm_client.generate.assert_called_once_with(
            "hello", max_new_tokens=64, temperature=0.3
        )

    @pytest.mark.asyncio
    async def test_backend_health_proxy(self):
        p = self._make_pipeline()
        result = await p.backend_health()
        assert result["reachable"] is True

    @pytest.mark.asyncio
    async def test_rate_limit_async(self):
        p = self._make_pipeline(enable_rate_limiting=True, max_requests_per_minute=1)
        await p.async_process("first", client_id="u1")
        status, report = await p.async_process("second", client_id="u1")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.RATE_LIMIT

    @pytest.mark.asyncio
    async def test_input_too_long_async(self):
        p = self._make_pipeline(max_prompt_length=5)
        status, report = await p.async_process("this is too long")
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.INPUT_TOO_LONG


# ═══════════════════════════════════════════════════════════════════════════
# Convenience helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_create_pipeline_default(self):
        p = create_pipeline()
        assert isinstance(p, SafetyPipeline)

    def test_create_pipeline_with_backend(self):
        bc = LLMBackendConfig(model="llama3")
        p = create_pipeline(backend_config=bc)
        assert p.backend_config.model == "llama3"

    def test_quick_check_safe(self):
        cfg = SafetyConfig(save_reports=False, log_level="ERROR",
                           enable_semantic_check=False)
        _pipeline_cache.clear()
        result = quick_check("What is quantum computing?", config=cfg)
        assert result["is_safe"] is True
        assert "safety_level" in result

    def test_quick_check_unsafe(self):
        cfg = SafetyConfig(save_reports=False, log_level="ERROR",
                           enable_semantic_check=False)
        _pipeline_cache.clear()
        result = quick_check("how to make explosives", config=cfg)
        assert result["is_safe"] is False

    def test_quick_check_reuses_pipeline(self):
        cfg = SafetyConfig(save_reports=False, log_level="ERROR",
                           enable_semantic_check=False)
        _pipeline_cache.clear()
        quick_check("hello", config=cfg)
        key_before = list(_pipeline_cache.keys())
        quick_check("world", config=cfg)
        assert list(_pipeline_cache.keys()) == key_before


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def _pipeline(self):
        return SafetyPipeline(config=SafetyConfig(
            save_reports=False,
            log_level="ERROR",
            enable_semantic_check=False,
        ))

    def test_empty_prompt(self):
        p = self._pipeline()
        status, _ = p.process("")
        assert status in ("ACCEPTED", "REJECTED")

    def test_very_long_prompt_rejected(self):
        p = SafetyPipeline(config=SafetyConfig(
            max_prompt_length=100,
            save_reports=False,
            log_level="ERROR",
            enable_semantic_check=False,
        ))
        status, report = p.process("x" * 200)
        assert status == "REJECTED"
        assert report.rejection_reason == RejectionReason.INPUT_TOO_LONG

    def test_special_characters_handled(self):
        p = self._pipeline()
        status, _ = p.process("Test: @#$%^&*()_+-=[]{}|;':\",./<>?")
        assert status in ("ACCEPTED", "REJECTED")

    @pytest.mark.asyncio
    async def test_async_process_with_empty_generation_kwargs(self):
        p = self._pipeline()
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=("result", 0.1))
        p.llm_client = mock_client
        status, _ = await p.async_process("hello", generation_kwargs={})
        assert status == "ACCEPTED"
        # Empty dict should pass no extra kwargs to generate.
        mock_client.generate.assert_called_once_with("hello")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
