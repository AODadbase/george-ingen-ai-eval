"""
Execution layer: async API clients for all four LLM providers.

Every LLMResponse records provider + model + latency_ms + token counts
per the SREGym benchmark standard (no result without full reproduction metadata).

Providers implemented:
  - OpenAIClient      → GPT-4o via openai async SDK
  - AnthropicClient   → Claude via anthropic async SDK
  - DeepSeekClient    → deepseek-chat via aiohttp (OpenAI-compatible REST)
  - HuggingFaceClient → Llama-3-8B-Instruct via HuggingFace Inference API (aiohttp)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import aiohttp
import anthropic
import openai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured result returned by every client call."""
    provider: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    temperature: float = 0.0
    seed: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider-agnostic exceptions
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Raised when a provider returns HTTP 429 or an SDK rate-limit signal."""


class TransientError(Exception):
    """Raised on HTTP 5xx or transient network errors that warrant a retry."""


# ---------------------------------------------------------------------------
# Abstract base client
# ---------------------------------------------------------------------------

class BaseAsyncLLMClient(ABC):
    """
    Abstract base for all provider clients.

    Subclasses implement _call(); this class adds:
      - asyncio.wait_for timeout enforcement
      - Exponential-backoff retry on RateLimitError and TransientError
      - Graceful error return (never raises to the caller)
    """

    MAX_RETRIES: int = 3
    BASE_BACKOFF_S: float = 2.0  # doubles each attempt: 2s, 4s, 8s

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout_s = timeout_s

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def _call(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...

    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Public entry point. Wraps _call() with timeout + retry logic.

        Returns an LLMResponse with error field set on terminal failure
        rather than raising, so a failed provider never crashes the dispatcher.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self._call(system_prompt, user_prompt),
                    timeout=self.timeout_s,
                )

            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "[%s] Timeout on attempt %d/%d (limit=%.1fs)",
                    self.provider_name, attempt, self.MAX_RETRIES, self.timeout_s,
                )
                # Timeouts retry immediately up to MAX_RETRIES; no backoff sleep.

            except RateLimitError as exc:
                last_error = exc
                wait = self.BASE_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] Rate limit on attempt %d/%d — backing off %.1fs",
                    self.provider_name, attempt, self.MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)

            except TransientError as exc:
                last_error = exc
                wait = self.BASE_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] Transient error on attempt %d/%d — backing off %.1fs: %s",
                    self.provider_name, attempt, self.MAX_RETRIES, wait, exc,
                )
                await asyncio.sleep(wait)

            except Exception as exc:  # noqa: BLE001
                # Non-retryable error (e.g. 404 invalid model, auth failure).
                # Return immediately — no retry, no crash.
                logger.error(
                    "[%s] Non-retryable error on attempt %d: %s",
                    self.provider_name, attempt, exc,
                )
                return LLMResponse(
                    provider=self.provider_name,
                    model=self.model,
                    content="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0.0,
                    temperature=self.temperature,
                    seed=self.seed,
                    error=f"{type(exc).__name__}: {exc}",
                )

        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            temperature=self.temperature,
            seed=self.seed,
            error=f"Failed after {self.MAX_RETRIES} attempts: {last_error}",
        )


# ---------------------------------------------------------------------------
# OpenAI client  (GPT-4o)
# ---------------------------------------------------------------------------

class OpenAIClient(BaseAsyncLLMClient):
    """GPT-4o via the official openai async SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        seed: int = 42,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(model, temperature, seed, timeout_s)
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def _call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            content=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Anthropic client  (Claude)
# ---------------------------------------------------------------------------

class AnthropicClient(BaseAsyncLLMClient):
    """Claude via the official anthropic async SDK."""

    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.0,
        seed: int = 42,
        timeout_s: float = 60.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(model, temperature, seed, timeout_s)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def _call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise

        latency_ms = (time.perf_counter() - t0) * 1000
        content = response.content[0].text if response.content else ""
        usage = response.usage
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            content=content,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# DeepSeek client  (OpenAI-compatible REST via aiohttp)
# ---------------------------------------------------------------------------

_DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


class DeepSeekClient(BaseAsyncLLMClient):
    """
    DeepSeek via its OpenAI-compatible chat/completions endpoint.
    Uses aiohttp directly — no vendor SDK dependency.

    A single ClientSession is created on first use and reused for all
    concurrent requests, preserving the underlying TCP connection pool
    and avoiding TIME_WAIT socket exhaustion under concurrent dispatch.
    Call aclose() when the client is no longer needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        seed: int = 42,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(model, temperature, seed, timeout_s)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it lazily on first call."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session

    async def aclose(self) -> None:
        """Explicitly close the shared session. Call after all requests are done."""
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def provider_name(self) -> str:
        return "deepseek"

    async def _call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        t0 = time.perf_counter()
        session = self._get_session()
        async with session.post(
            f"{_DEEPSEEK_API_BASE}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
        ) as resp:
            body = await resp.text()
            if resp.status == 429:
                raise RateLimitError(f"DeepSeek 429: {body}")
            if resp.status >= 500:
                raise TransientError(f"DeepSeek {resp.status}: {body}")
            if resp.status != 200:
                raise RuntimeError(f"DeepSeek {resp.status}: {body}")
            data = await resp.json(content_type=None)

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = data.get("usage", {})
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            content=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# HuggingFace client  (Llama-3-8B-Instruct via Inference Router API)
# ---------------------------------------------------------------------------

# New Inference Providers Router — OpenAI-compatible endpoint.
# The legacy https://api-inference.huggingface.co/models/{model} endpoint
# Groq Cloud — OpenAI-compatible inference endpoint.
# The HF_TOKEN env variable is reused as the Groq API key for zero-.env changes.
_GROQ_API_BASE = "https://api.groq.com/openai/v1"


class HuggingFaceClient(BaseAsyncLLMClient):
    """
    Llama-3-8B via Groq Cloud (OpenAI-compatible endpoint).

    Groq's API is a drop-in replacement for HuggingFace Inference Providers:
    same chat/completions payload, same response shape, real token usage.
    The class retains the name HuggingFaceClient and reads from HF_TOKEN
    so no .env or dispatcher changes are required.

    A single shared ClientSession is reused across all concurrent requests
    to preserve the TCP connection pool.
    Call aclose() when the client is no longer needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        seed: int = 42,
        timeout_s: float = 90.0,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(model, temperature, seed, timeout_s)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.max_new_tokens = max_new_tokens
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it lazily on first call."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session

    async def aclose(self) -> None:
        """Explicitly close the shared session. Call after all requests are done."""
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def provider_name(self) -> str:
        return "groq"

    async def _call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Groq requires temperature > 0 when seed is set
            "temperature": max(self.temperature, 1e-4),
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
        }
        t0 = time.perf_counter()
        session = self._get_session()
        async with session.post(
            f"{_GROQ_API_BASE}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
        ) as resp:
            body = await resp.text()
            if resp.status == 429:
                raise RateLimitError(f"Groq 429: {body}")
            if resp.status >= 500:
                raise TransientError(f"Groq {resp.status}: {body}")
            if resp.status != 200:
                raise RuntimeError(f"Groq {resp.status}: {body}")
            data = await resp.json(content_type=None)

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = data.get("usage", {})
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            content=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
        )
