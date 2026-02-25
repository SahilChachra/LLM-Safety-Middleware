"""
LLM Safety Pipeline — Middle-Layer Proxy
=========================================

Acts as a safety middleware between clients and a remote LLM backend
(Ollama, OpenAI-compatible API, or any custom endpoint).

Request flow:
    Client → [Input Safety Checks] → External LLM (async HTTP) →
    [Output Safety Checks] → Client

Supported backends:
    - Ollama       (backend_type="ollama")
    - OpenAI-compatible (backend_type="openai")   — LM Studio, vLLM, Together, etc.
    - Custom       (backend_type="custom")

Version: 2.0.0
"""

import asyncio
import copy
import sys
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from enum import Enum
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import threading

import httpx
import torch
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _make_logger(
    name: str,
    level: int,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(_LOG_FORMATTER)
        logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(_LOG_FORMATTER)
        logger.addHandler(fh)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class SafetyLevel(Enum):
    SAFE = 0
    LOW_RISK = 1
    MEDIUM_RISK = 2
    HIGH_RISK = 3
    CRITICAL = 4


class RejectionReason(Enum):
    TOKEN_FILTER = "token_level_filter"
    PATTERN_MATCH = "rule_engine_pattern"
    SEMANTIC_UNSAFE = "semantic_classifier"
    POST_GEN_UNSAFE = "post_generation_check"
    RATE_LIMIT = "rate_limit_exceeded"
    MODEL_ERROR = "model_error"
    CUSTOM_FILTER = "custom_filter"
    INPUT_TOO_LONG = "input_too_long"
    BACKEND_UNAVAILABLE = "backend_unavailable"


@dataclass
class SafetyConfig:
    """Safety layer configuration (models, thresholds, filters)."""

    # Local safety models
    safety_model_name: str = "unitary/toxic-bert"
    spacy_model: str = "en_core_web_sm"

    # Thresholds
    safety_threshold: float = 0.75
    toxicity_threshold: float = 0.7

    # Which layers to enable
    enable_pattern_matching: bool = True
    enable_token_filtering: bool = True
    enable_semantic_check: bool = True
    enable_post_generation_check: bool = True

    # Device for local transformer inference (toxic-bert)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Input guardrail
    max_prompt_length: int = 10_000

    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60

    # Logging & reporting
    log_level: str = "INFO"
    log_file: Optional[str] = None
    save_reports: bool = True
    reports_dir: str = "./safety_reports"

    # Extensible pattern lists
    custom_banned_patterns: List[str] = field(default_factory=list)
    custom_allowed_contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetyConfig":
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "SafetyConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class LLMBackendConfig:
    """
    Configuration for the remote LLM backend.

    Supports Ollama, any OpenAI-compatible REST API, or a fully
    custom endpoint.  Per-request overrides (temperature, top_p, etc.)
    can be passed via generation_kwargs to async_process().
    """

    # Protocol: "ollama" | "openai" | "custom"
    backend_type: str = "ollama"

    # Remote LLM server
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    api_key: Optional[str] = None  # Bearer token for OpenAI-compatible endpoints

    # Connection / resilience
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Generation defaults (overridable per-request)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    system_prompt: Optional[str] = None  # Prepended as system message (OpenAI mode)

    # Custom backend wire format
    custom_endpoint: str = "/api/generate"
    custom_prompt_field: str = "prompt"
    custom_response_field: str = "response"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMBackendConfig":
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_env(cls) -> "LLMBackendConfig":
        """Construct from environment variables (useful in containerised deployments)."""
        import os
        return cls(
            backend_type=os.environ.get("LLM_BACKEND_TYPE", "ollama"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("LLM_MODEL", "llama2"),
            api_key=os.environ.get("LLM_API_KEY"),
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "512")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("LLM_TOP_P", "0.95")),
            system_prompt=os.environ.get("LLM_SYSTEM_PROMPT"),
        )


@dataclass
class SafetyReport:
    """Per-request record of every safety decision made."""

    timestamp: str
    prompt: str
    status: str                        # "ACCEPTED" | "REJECTED" | "PENDING"
    safety_level: SafetyLevel
    rejection_reason: Optional[RejectionReason] = None

    # None = check was disabled in config
    token_filter_passed: Optional[bool] = None
    pattern_check_passed: Optional[bool] = None
    semantic_check_passed: Optional[bool] = None
    post_gen_check_passed: Optional[bool] = None

    safety_score: Optional[float] = None
    toxicity_score: Optional[float] = None

    matched_patterns: List[str] = field(default_factory=list)
    flagged_tokens: List[str] = field(default_factory=list)

    generated_text: Optional[str] = None
    generation_time: Optional[float] = None

    model_version: Optional[str] = None
    config_snapshot: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["safety_level"] = self.safety_level.name
        if self.rejection_reason:
            data["rejection_reason"] = self.rejection_reason.value
        return data

    def save(self, directory: str) -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        fpath = Path(directory) / fname
        with open(fpath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(fpath)


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

class PatternRegistry:
    """Regex pattern sets for harmful and safe contexts."""

    HARMFUL_INSTRUCTION_PATTERNS = [
        r"\b(?:how\s+to|guide\s+to|steps\s+to|ways\s+to)\s+(?:kill|murder|assassinate|harm|hurt|attack)",
        r"\b(?:plan|planning|create|make)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison)",
        r"\bterrorist\s+attack\b",
        r"\b(?:how\s+to|ways\s+to)\s+(?:commit\s+)?suicide\b",
        r"\bself[-\s]harm\s+(?:methods|techniques|guide)",
        r"\b(?:how\s+to|guide\s+to)\s+(?:hack|crack|break\s+into|steal)",
        r"\b(?:make|create|produce)\s+(?:fake|counterfeit)\s+(?:id|passport|money)",
        r"\bdrug\s+(?:production|manufacturing|synthesis)\b",
        r"\b(?:child|minor)\s+(?:exploitation|abuse|grooming)",
        r"\bhuman\s+trafficking\b",
        r"\b(?:bioweapon|biological\s+weapon)\s+(?:creation|development)",
        r"\bchemical\s+weapon\b",
    ]

    # Presence of these shifts control to the semantic classifier; they do NOT
    # bypass safety checks.
    SAFE_CONTEXT_PATTERNS = [
        r"\b(?:history|historical|documentary|educational)\b",
        r"\b(?:novel|fiction|movie|film|screenplay|story)\b",
        r"\b(?:prevent|prevention|stop|avoid|protect)\b",
        r"\b(?:research|study|analysis|understand)\b",
        r"\b(?:mental\s+health|therapy|counseling|support)\b",
    ]

    CONTEXT_SENSITIVE_TOKENS = {
        "kill":   ["how to", "plan", "guide to", "steps to"],
        "bomb":   ["make", "create", "build", "construct"],
        "hack":   ["how to", "guide", "tutorial"],
        "poison": ["make", "create", "produce"],
    }

    @classmethod
    def compile_patterns(cls, extra: Optional[List[str]] = None) -> "re.Pattern":
        import re
        patterns = cls.HARMFUL_INSTRUCTION_PATTERNS + (extra or [])
        return re.compile("|".join(patterns), flags=re.IGNORECASE)

    @classmethod
    def compile_safe_patterns(cls, extra: Optional[List[str]] = None) -> "re.Pattern":
        import re
        patterns = cls.SAFE_CONTEXT_PATTERNS + (extra or [])
        return re.compile("|".join(patterns), flags=re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Thread-safe per-client sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str = "default") -> bool:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]
            if len(self._requests[client_id]) < self.max_requests:
                self._requests[client_id].append(now)
                return True
            return False

    def reset(self, client_id: str = "default") -> None:
        with self._lock:
            self._requests[client_id] = []


# ═══════════════════════════════════════════════════════════════════════════
# SAFETY FILTERS
# ═══════════════════════════════════════════════════════════════════════════

class TokenLevelFilter:
    """Regex + context-sensitive token safety filter."""

    def __init__(self, patterns: PatternRegistry, logger: logging.Logger):
        self.patterns = patterns
        self.logger = logger
        self.harmful_regex = patterns.compile_patterns()
        self.safe_regex = patterns.compile_safe_patterns()

    def check(self, text: str) -> Tuple[bool, List[str], List[str]]:
        """
        Returns (is_safe, matched_harmful_patterns, flagged_context_tokens).

        When harmful patterns are found alongside safe-context keywords, the
        filter returns True (passing) but preserves matched_patterns so the
        semantic classifier can make the final call.
        """
        import re
        text_lower = text.lower()
        has_safe_ctx = bool(self.safe_regex.search(text_lower))

        matched = [m.group(0) for m in self.harmful_regex.finditer(text_lower)]
        if matched:
            if has_safe_ctx:
                self.logger.warning(
                    f"Harmful patterns with safe-context keywords {matched!r} — "
                    "deferring to semantic classifier."
                )
                return True, matched, []
            self.logger.warning(f"Harmful patterns: {matched!r}")
            return False, matched, []

        if not has_safe_ctx:
            flagged: List[str] = []
            for token, contexts in self.patterns.CONTEXT_SENSITIVE_TOKENS.items():
                if token in text_lower:
                    for ctx in contexts:
                        if ctx in text_lower:
                            flagged.append(f"{token} (context: {ctx})")
                            self.logger.warning(
                                f"Dangerous token-context pair: '{token}' with '{ctx}'"
                            )
            if flagged:
                return False, [], flagged

        return True, [], []


class RuleEngineFilter:
    """NLP-based rule engine using spaCy dependency parsing."""

    def __init__(self, spacy_model: str, logger: logging.Logger):
        self.logger = logger
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.logger.warning(f"spaCy model '{spacy_model}' not found — downloading…")
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", spacy_model],
                check=False,
            )
            self.nlp = spacy.load(spacy_model)

    def check(self, text: str) -> Tuple[bool, List[str]]:
        doc = self.nlp(text.lower())
        reasons: List[str] = []

        if self._has_harmful_instruction_pattern(doc):
            reasons.append("Harmful instruction pattern detected")
        if self._has_dangerous_verb_object(doc):
            reasons.append("Dangerous action-target combination detected")
        if self._has_prohibited_entities(doc):
            reasons.append("Prohibited entity referenced")

        if reasons:
            self.logger.warning(f"Rule engine flagged: {reasons}")
        return len(reasons) == 0, reasons

    def _has_harmful_instruction_pattern(self, doc) -> bool:
        harmful_verbs = {
            "kill", "murder", "bomb", "attack", "harm", "poison", "hack",
            "synthesize", "detonate", "assassinate", "exploit", "stalk", "traffic",
        }
        instruction_triggers = {
            "how", "steps", "guide", "plan", "way", "method",
            "instruction", "tutorial", "procedure", "process",
        }
        for token in doc:
            if token.lemma_ not in harmful_verbs or token.pos_ != "VERB":
                continue
            for ancestor in token.ancestors:
                if ancestor.lemma_ in instruction_triggers:
                    return True
            for sibling in token.head.children:
                if sibling.lemma_ in instruction_triggers:
                    return True
            for child in token.children:
                if child.lemma_ in instruction_triggers:
                    return True
        # Positional fallback for the simplest patterns.
        for i, token in enumerate(doc):
            if token.text in instruction_triggers and i + 2 < len(doc):
                if doc[i + 1].text == "to" and doc[i + 2].lemma_ in harmful_verbs:
                    return True
        return False

    def _has_dangerous_verb_object(self, doc) -> bool:
        dangerous = {"kill", "murder", "harm", "attack", "exploit"}
        targets = {"child", "minor", "person", "people", "civilian"}
        for token in doc:
            if token.lemma_ in dangerous:
                for child in token.children:
                    if child.dep_ == "dobj" and child.lemma_ in targets:
                        return True
        return False

    def _has_prohibited_entities(self, doc) -> bool:
        prohibited = {"terrorist", "extremist", "jihadi", "jihadist", "white supremacist", "neo-nazi"}
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "NORP"):
                if any(t in ent.text.lower() for t in prohibited):
                    return True
        return any(t in doc.text.lower() for t in prohibited)


class SemanticSafetyClassifier:
    """Transformer-based toxicity classifier (supports binary and multi-label)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        safety_threshold: float,
        toxicity_threshold: float,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.device = device
        self.safety_threshold = safety_threshold
        self.toxicity_threshold = toxicity_threshold

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self.logger.info(f"Loaded safety classifier: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load safety model: {e}")
            raise

    def check(self, text: str) -> Tuple[bool, float, SafetyLevel]:
        """
        Returns (is_safe, safe_probability, safety_level).
        Fails closed on any error.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze()

            num_labels = logits.shape[-1] if logits.dim() > 0 else 1
            if num_labels == 2:
                # Binary: index 0 = safe, index 1 = toxic.
                probs = torch.softmax(logits, dim=-1)
                toxic_prob = probs[1].item()
            else:
                # Multi-label (e.g. unitary/toxic-bert with 6 heads).
                # Use sigmoid + worst-case score across all categories.
                probs = torch.sigmoid(logits)
                toxic_prob = probs.max().item()

            safe_prob = 1.0 - toxic_prob

            if toxic_prob < 0.2:
                level = SafetyLevel.SAFE
            elif toxic_prob < 0.4:
                level = SafetyLevel.LOW_RISK
            elif toxic_prob < 0.6:
                level = SafetyLevel.MEDIUM_RISK
            elif toxic_prob < 0.8:
                level = SafetyLevel.HIGH_RISK
            else:
                level = SafetyLevel.CRITICAL

            is_safe = safe_prob >= self.safety_threshold
            self.logger.debug(
                f"Semantic — safe: {safe_prob:.3f}, toxic: {toxic_prob:.3f}, "
                f"level: {level.name}"
            )
            return is_safe, safe_prob, level

        except Exception as e:
            self.logger.error(f"Semantic classifier error: {e}")
            return False, 0.0, SafetyLevel.CRITICAL  # fail closed


# ═══════════════════════════════════════════════════════════════════════════
# EXTERNAL LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class ExternalLLMClient:
    """
    Async HTTP client that forwards safe prompts to a remote LLM and returns
    the generated text.

    Supports three protocols:
        "ollama"  — Ollama REST API  (POST /api/generate)
        "openai"  — OpenAI-compatible (POST /v1/chat/completions)
        "custom"  — Configurable endpoint and payload field names
    """

    def __init__(self, config: LLMBackendConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
            headers=headers,
        )

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Call the remote LLM with exponential-backoff retry on transient errors.

        Returns:
            (response_text, elapsed_seconds)

        Raises:
            RuntimeError on non-retryable errors or exhausted retries.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await self._dispatch(prompt, max_new_tokens, temperature, top_p)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code < 500:
                    raise RuntimeError(
                        f"LLM backend rejected the request "
                        f"(HTTP {exc.response.status_code}): {exc.response.text}"
                    ) from exc
                self.logger.warning(
                    f"LLM backend error {exc.response.status_code} "
                    f"(attempt {attempt}/{self.config.max_retries})"
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as exc:
                last_exc = exc
                self.logger.warning(
                    f"LLM backend connection error "
                    f"(attempt {attempt}/{self.config.max_retries}): {exc}"
                )
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay_seconds * attempt)

        raise RuntimeError(
            f"LLM backend unavailable after {self.config.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Probe the backend; returns a status dict."""
        t = self.config.backend_type.lower()
        base = self.config.base_url.rstrip("/")
        if t == "ollama":
            probe = f"{base}/api/tags"
        elif t == "openai":
            probe = f"{base}/v1/models"
        else:
            probe = base
        try:
            resp = await self._client.get(probe, timeout=5.0)
            reachable = resp.status_code < 500
        except Exception as exc:
            self.logger.warning(f"Backend health probe failed: {exc}")
            reachable = False
        return {
            "reachable": reachable,
            "backend_type": self.config.backend_type,
            "base_url": self.config.base_url,
            "model": self.config.model,
        }

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    #  Backend-specific implementations                                    #
    # ------------------------------------------------------------------ #

    async def _dispatch(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Tuple[str, float]:
        t = self.config.backend_type.lower()
        if t == "ollama":
            return await self._call_ollama(prompt, max_new_tokens, temperature, top_p)
        if t == "openai":
            return await self._call_openai(prompt, max_new_tokens, temperature, top_p)
        if t == "custom":
            return await self._call_custom(prompt, max_new_tokens, temperature, top_p)
        raise ValueError(
            f"Unknown backend_type: {t!r}. Choose 'ollama', 'openai', or 'custom'."
        )

    async def _call_ollama(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Tuple[str, float]:
        options: Dict[str, Any] = {}
        eff = max_new_tokens or self.config.max_new_tokens
        if eff is not None:
            options["num_predict"] = eff
        t = temperature if temperature is not None else self.config.temperature
        if t is not None:
            options["temperature"] = t
        p = top_p if top_p is not None else self.config.top_p
        if p is not None:
            options["top_p"] = p

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        t0 = time.monotonic()
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        elapsed = time.monotonic() - t0

        text: str = resp.json().get("response", "")
        self.logger.debug(
            f"Ollama {self.config.model!r}: {len(text)} chars in {elapsed:.2f}s"
        )
        return text, elapsed

    async def _call_openai(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Tuple[str, float]:
        messages: List[Dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {"model": self.config.model, "messages": messages}
        eff = max_new_tokens or self.config.max_new_tokens
        if eff is not None:
            payload["max_tokens"] = eff
        t = temperature if temperature is not None else self.config.temperature
        if t is not None:
            payload["temperature"] = t
        p = top_p if top_p is not None else self.config.top_p
        if p is not None:
            payload["top_p"] = p

        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        t0 = time.monotonic()
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        elapsed = time.monotonic() - t0

        data = resp.json()
        text: str = data["choices"][0]["message"]["content"]
        self.logger.debug(
            f"OpenAI-compat {self.config.model!r}: {len(text)} chars in {elapsed:.2f}s"
        )
        return text, elapsed

    async def _call_custom(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Tuple[str, float]:
        payload: Dict[str, Any] = {self.config.custom_prompt_field: prompt}
        eff = max_new_tokens or self.config.max_new_tokens
        if eff is not None:
            payload["max_tokens"] = eff
        t = temperature if temperature is not None else self.config.temperature
        if t is not None:
            payload["temperature"] = t

        endpoint = self.config.custom_endpoint.lstrip("/")
        url = f"{self.config.base_url.rstrip('/')}/{endpoint}"
        t0 = time.monotonic()
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        elapsed = time.monotonic() - t0

        text: str = resp.json().get(self.config.custom_response_field, "")
        self.logger.debug(f"Custom backend: {len(text)} chars in {elapsed:.2f}s")
        return text, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SAFETY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class SafetyPipeline:
    """
    Middle-layer safety proxy.

    process()       — synchronous input-only safety check (used by /check)
    async_process() — full async pipeline: input check → remote LLM → output check
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        backend_config: Optional[LLMBackendConfig] = None,
    ):
        self.config = config or SafetyConfig()
        self.backend_config = backend_config or LLMBackendConfig()

        self.logger = _make_logger(
            name=f"llm_safety.pipeline.{id(self)}",
            level=getattr(logging, self.config.log_level.upper()),
            log_file=self.config.log_file,
        )
        self.logger.info("Initializing Safety Pipeline…")

        self._custom_filters: List[Tuple[str, Callable[[str], Tuple[bool, str]]]] = []
        self._config_snapshot: Dict[str, Any] = {
            "safety": self.config.to_dict(),
            "backend": {
                k: v for k, v in self.backend_config.to_dict().items()
                if k != "api_key"   # Never persist the API key in reports.
            },
        }

        self._initialize_components()

        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_reasons": defaultdict(int),
            "avg_processing_time": 0.0,
        }
        self._stats_lock = threading.Lock()
        self.logger.info("Safety Pipeline ready.")

    # ------------------------------------------------------------------ #
    #  Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _initialize_components(self) -> None:
        self.rate_limiter = (
            RateLimiter(
                max_requests=self.config.max_requests_per_minute,
                window_seconds=60,
            )
            if self.config.enable_rate_limiting
            else None
        )

        self.pattern_registry = PatternRegistry()
        self._cached_harmful_regex = self.pattern_registry.compile_patterns(
            self.config.custom_banned_patterns or None
        )

        self.token_filter: Optional[TokenLevelFilter] = None
        if self.config.enable_token_filtering:
            self.token_filter = TokenLevelFilter(
                patterns=self.pattern_registry,
                logger=self.logger,
            )

        self.rule_engine: Optional[RuleEngineFilter] = None
        if self.config.enable_pattern_matching:
            self.logger.info("Loading spaCy rule engine…")
            self.rule_engine = RuleEngineFilter(
                spacy_model=self.config.spacy_model,
                logger=self.logger,
            )

        self.semantic_classifier: Optional[SemanticSafetyClassifier] = None
        if self.config.enable_semantic_check:
            self.logger.info("Loading semantic safety classifier…")
            self.semantic_classifier = SemanticSafetyClassifier(
                model_name=self.config.safety_model_name,
                device=self.config.device,
                safety_threshold=self.config.safety_threshold,
                toxicity_threshold=self.config.toxicity_threshold,
                logger=self.logger,
            )

        self.llm_client = ExternalLLMClient(
            config=self.backend_config,
            logger=self.logger,
        )
        self.logger.info(
            f"External LLM client configured: "
            f"{self.backend_config.backend_type} @ {self.backend_config.base_url} "
            f"model={self.backend_config.model!r}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def process(
        self,
        prompt: str,
        client_id: str = "default",
    ) -> Tuple[str, "SafetyReport"]:
        """
        Synchronous input-only safety check.

        Runs all configured input safety layers and returns immediately
        without calling the external LLM.  Use this for the /check endpoint
        or any situation where you want to pre-validate a prompt.

        Returns:
            ("ACCEPTED" | "REJECTED", SafetyReport)
        """
        t0 = time.monotonic()
        report = self._make_report(prompt)
        try:
            if not self._check_input_safety(prompt, report, client_id):
                self._update_stats(report, time.monotonic() - t0)
                return "REJECTED", report
            report.status = "ACCEPTED"
            self._update_stats(report, time.monotonic() - t0)
            if self.config.save_reports:
                report.save(self.config.reports_dir)
            return "ACCEPTED", report
        except Exception as e:
            self.logger.error(f"process() error: {e}", exc_info=True)
            report.status = "REJECTED"
            report.rejection_reason = RejectionReason.MODEL_ERROR
            report.safety_level = SafetyLevel.CRITICAL
            self._update_stats(report, time.monotonic() - t0)
            return "REJECTED", report

    async def async_process(
        self,
        prompt: str,
        client_id: str = "default",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, "SafetyReport"]:
        """
        Full async pipeline: input safety → remote LLM → output safety.

        CPU-bound safety checks are offloaded to a thread-pool executor so
        they never block the event loop.  The external LLM call is a true
        async HTTP request.

        Args:
            prompt:            User prompt.
            client_id:         Per-client rate-limit identifier.
            generation_kwargs: Optional overrides forwarded to the LLM:
                               max_new_tokens, temperature, top_p.

        Returns:
            ("ACCEPTED" | "REJECTED", SafetyReport)
        """
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        report = self._make_report(prompt)

        try:
            # ── Step 1: Input safety checks (sync → thread pool) ─────────
            input_safe = await loop.run_in_executor(
                None, self._check_input_safety, prompt, report, client_id
            )
            if not input_safe:
                self._update_stats(report, time.monotonic() - t0)
                return "REJECTED", report

            # ── Step 2: External LLM call (async HTTP) ────────────────────
            gen_kwargs = generation_kwargs or {}
            try:
                generated_text, gen_time = await self.llm_client.generate(
                    prompt, **gen_kwargs
                )
            except RuntimeError as exc:
                self.logger.error(f"External LLM failed: {exc}")
                report.status = "REJECTED"
                report.rejection_reason = RejectionReason.BACKEND_UNAVAILABLE
                report.safety_level = SafetyLevel.CRITICAL
                self._update_stats(report, time.monotonic() - t0)
                return "REJECTED", report

            report.generated_text = generated_text
            report.generation_time = gen_time

            # ── Step 3: Output safety check (sync → thread pool) ──────────
            if self.config.enable_post_generation_check:
                output_safe = await loop.run_in_executor(
                    None, self._post_generation_check, generated_text, report
                )
                if not output_safe:
                    self.logger.warning("Post-generation safety check failed.")
                    report.status = "REJECTED"
                    report.rejection_reason = RejectionReason.POST_GEN_UNSAFE
                    report.safety_level = SafetyLevel.MEDIUM_RISK
                    report.generated_text = "[Content filtered for safety]"
                    self._update_stats(report, time.monotonic() - t0)
                    return "REJECTED", report

            # ── All checks passed ─────────────────────────────────────────
            report.status = "ACCEPTED"
            self._update_stats(report, time.monotonic() - t0)
            if self.config.save_reports:
                await loop.run_in_executor(
                    None, report.save, self.config.reports_dir
                )
            return "ACCEPTED", report

        except Exception as e:
            self.logger.error(f"async_process() error: {e}", exc_info=True)
            report.status = "REJECTED"
            report.rejection_reason = RejectionReason.MODEL_ERROR
            report.safety_level = SafetyLevel.CRITICAL
            self._update_stats(report, time.monotonic() - t0)
            return "REJECTED", report

    async def backend_health(self) -> Dict[str, Any]:
        """Probe the configured LLM backend and return a status dict."""
        return await self.llm_client.health_check()

    async def async_close(self) -> None:
        """Release the underlying HTTP connection pool. Call on shutdown."""
        await self.llm_client.close()
        self.logger.info("Safety Pipeline shut down.")

    def add_custom_filter(
        self,
        name: str,
        filter_func: Callable[[str], Tuple[bool, str]],
    ) -> None:
        """
        Register a custom synchronous filter.

        filter_func(prompt) → (is_safe: bool, reason: str)
        Raises ValueError if a filter with the same name already exists.
        """
        if any(n == name for n, _ in self._custom_filters):
            raise ValueError(f"Custom filter '{name}' is already registered.")
        self._custom_filters.append((name, filter_func))
        self.logger.info(f"Custom filter '{name}' registered.")

    def remove_custom_filter(self, name: str) -> bool:
        """Remove a custom filter by name. Returns True if found."""
        before = len(self._custom_filters)
        self._custom_filters = [(n, f) for n, f in self._custom_filters if n != name]
        return len(self._custom_filters) < before

    def get_statistics(self) -> Dict[str, Any]:
        """Return a deep copy of current statistics (thread-safe)."""
        with self._stats_lock:
            return copy.deepcopy(dict(self.stats))

    def reset_statistics(self) -> None:
        with self._stats_lock:
            self.stats = {
                "total_requests": 0,
                "accepted": 0,
                "rejected": 0,
                "rejection_reasons": defaultdict(int),
                "avg_processing_time": 0.0,
            }

    # ------------------------------------------------------------------ #
    #  Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "SafetyPipeline":
        return self

    def __exit__(self, *_) -> bool:
        # Synchronous cleanup only.  For full cleanup call async_close().
        self.logger.info("SafetyPipeline context exited.")
        return False

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_report(self, prompt: str) -> SafetyReport:
        return SafetyReport(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            status="PENDING",
            safety_level=SafetyLevel.SAFE,
            model_version=self.backend_config.model,
            config_snapshot=self._config_snapshot,
        )

    def _check_input_safety(
        self,
        prompt: str,
        report: SafetyReport,
        client_id: str,
    ) -> bool:
        """
        Run all input safety layers and mutate report in-place.
        Returns True if the prompt is safe to forward to the LLM.
        Safe to call from a thread-pool executor.
        """
        # 1. Length guard
        if len(prompt) > self.config.max_prompt_length:
            self.logger.warning(
                f"Prompt length {len(prompt)} exceeds limit "
                f"{self.config.max_prompt_length}"
            )
            report.status = "REJECTED"
            report.rejection_reason = RejectionReason.INPUT_TOO_LONG
            report.safety_level = SafetyLevel.HIGH_RISK
            return False

        # 2. Rate limiting
        if self.rate_limiter and not self.rate_limiter.is_allowed(client_id):
            self.logger.warning(f"Rate limit exceeded: {client_id!r}")
            report.status = "REJECTED"
            report.rejection_reason = RejectionReason.RATE_LIMIT
            report.safety_level = SafetyLevel.HIGH_RISK
            return False

        # 3. Token filter
        if self.token_filter:
            is_safe, matched, flagged = self.token_filter.check(prompt)
            report.token_filter_passed = is_safe
            report.matched_patterns = matched
            report.flagged_tokens = flagged
            if not is_safe:
                report.status = "REJECTED"
                report.rejection_reason = RejectionReason.TOKEN_FILTER
                report.safety_level = SafetyLevel.HIGH_RISK
                return False

        # 4. Rule engine
        if self.rule_engine:
            is_safe, reasons = self.rule_engine.check(prompt)
            report.pattern_check_passed = is_safe
            if not is_safe:
                report.status = "REJECTED"
                report.rejection_reason = RejectionReason.PATTERN_MATCH
                report.matched_patterns.extend(reasons)
                report.safety_level = SafetyLevel.HIGH_RISK
                return False

        # 5. Custom filters
        for fname, ffunc in self._custom_filters:
            try:
                is_safe, reason = ffunc(prompt)
            except Exception as exc:
                self.logger.error(f"Custom filter '{fname}' error: {exc}", exc_info=True)
                is_safe, reason = False, f"filter error: {exc}"
            if not is_safe:
                report.status = "REJECTED"
                report.rejection_reason = RejectionReason.CUSTOM_FILTER
                report.matched_patterns.append(f"[{fname}] {reason}")
                report.safety_level = SafetyLevel.HIGH_RISK
                return False

        # 6. Semantic classifier
        if self.semantic_classifier:
            is_safe, score, level = self.semantic_classifier.check(prompt)
            report.semantic_check_passed = is_safe
            report.safety_score = score
            report.safety_level = level
            if not is_safe:
                report.status = "REJECTED"
                report.rejection_reason = RejectionReason.SEMANTIC_UNSAFE
                return False

        return True

    def _post_generation_check(self, text: str, report: SafetyReport) -> bool:
        """
        Validate the LLM's output through the same safety layers.
        Safe to call from a thread-pool executor.
        """
        if self.token_filter:
            is_safe, _, _ = self.token_filter.check(text)
            if not is_safe:
                report.post_gen_check_passed = False
                return False

        if self._cached_harmful_regex.search(text.lower()):
            report.post_gen_check_passed = False
            return False

        if self.semantic_classifier:
            is_safe, _, _ = self.semantic_classifier.check(text)
            if not is_safe:
                report.post_gen_check_passed = False
                return False

        report.post_gen_check_passed = True
        return True

    def _update_stats(self, report: SafetyReport, elapsed: float) -> None:
        with self._stats_lock:
            self.stats["total_requests"] += 1
            if report.status == "ACCEPTED":
                self.stats["accepted"] += 1
            else:
                self.stats["rejected"] += 1
                if report.rejection_reason:
                    self.stats["rejection_reasons"][report.rejection_reason.value] += 1
            n = self.stats["total_requests"]
            old = self.stats["avg_processing_time"]
            self.stats["avg_processing_time"] = (old * (n - 1) + elapsed) / n


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_pipeline(
    config_path: Optional[str] = None,
    backend_config: Optional[LLMBackendConfig] = None,
) -> SafetyPipeline:
    """
    Create a SafetyPipeline from an optional JSON config file.

    The backend config can be supplied directly or will default to
    LLMBackendConfig() (Ollama on localhost).
    """
    config = SafetyConfig.from_json(config_path) if config_path else SafetyConfig()
    return SafetyPipeline(config=config, backend_config=backend_config)


# Module-level cache: quick_check() reuses the same pipeline across calls.
_pipeline_cache: Dict[str, SafetyPipeline] = {}


def quick_check(prompt: str, config: Optional[SafetyConfig] = None) -> Dict[str, Any]:
    """
    Input-only safety check with a cached pipeline.

    The pipeline (which loads spaCy + toxic-bert) is created once per
    unique config and reused on subsequent calls.

    Returns:
        {"is_safe": bool, "safety_level": str, "safety_score": float|None,
         "reasons": list[str]}
    """
    cache_key = "__default__" if config is None else json.dumps(
        config.to_dict(), sort_keys=True
    )
    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = SafetyPipeline(config=config or SafetyConfig())

    status, report = _pipeline_cache[cache_key].process(prompt)
    return {
        "is_safe": status == "ACCEPTED",
        "safety_level": report.safety_level.name,
        "safety_score": report.safety_score,
        "reasons": report.matched_patterns,
    }


if __name__ == "__main__":
    pass
