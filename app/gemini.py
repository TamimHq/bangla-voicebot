"""Shared Gemini request settings.

Gemini 3.x Flash models "think" before answering, and those hidden thinking
tokens count against max_output_tokens. With a small cap the model can spend
the whole budget thinking and return an empty or cut-off answer. For a voice
bot we want almost no thinking (it adds latency too), so we ask for the lowest
thinking level and leave plenty of output headroom.
"""
from __future__ import annotations

import logging
import re
import time

from .config import settings

log = logging.getLogger("voicebot")
_thinking_supported = True


def make_client():
    """Gemini client with a short, explicit retry policy.

    Some SDK versions retry 429/503 errors up to 5 times with backoff of up to
    60 s, so an overloaded or rate-limited API looks like a silent 30 s hang.
    For a phone call it's better to retry once quickly and then fail visibly.
    """
    from google import genai
    from google.genai import types

    opts = types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=2, initial_delay=0.5, max_delay=2.0,
            # Not 429: retrying a quota error just burns time and more quota.
            # Quota errors are handled by switching model (see ModelPool).
            http_status_codes=[500, 502, 503, 504],
        )
    )
    return genai.Client(api_key=settings.gemini_api_key, http_options=opts)


# ---------------- free-tier quota handling ----------------
# The free tier allows only a few requests per minute, counted PER MODEL. When
# one model returns 429, we park it for the delay Google asks for and use the
# next model in the list right away, instead of waiting.

class QuotaExhausted(RuntimeError):
    def __init__(self, wait_s: float):
        self.wait_s = wait_s
        super().__init__(f"Gemini free-tier limit reached on every model; try again in {wait_s:.0f}s")


_parked_until: dict[str, float] = {}  # model -> time.monotonic() when usable again


def is_quota_error(err: Exception) -> bool:
    code = getattr(err, "code", None)
    return code == 429 or "RESOURCE_EXHAUSTED" in str(err)


def retry_delay_s(err: Exception, default: float = 60.0) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(err)) or re.search(r"'retryDelay': '(\d+)s'", str(err))
    return float(m.group(1)) if m else default


def candidate_models(primary: str) -> list[str]:
    """primary first, then LLM_FALLBACK_MODELS, without duplicates."""
    out = []
    for name in [primary, *settings.llm_fallback_models]:
        if name and name not in out:
            out.append(name)
    return out


def usable_models(primary: str) -> list[str]:
    now = time.monotonic()
    models = candidate_models(primary)
    free = [mdl for mdl in models if _parked_until.get(mdl, 0) <= now]
    if not free:
        raise QuotaExhausted(min(_parked_until[mdl] for mdl in models) - now)
    return free


def quota_kind(err: Exception) -> str:
    """'per-day', 'per-minute' or 'unknown', from the quotaId in Google's 429 body."""
    m = re.search(r"quotaId['\"]?\s*[:=]\s*['\"]?([A-Za-z-]+)", str(err))
    qid = m.group(1) if m else ""
    if "PerDay" in qid:
        return "per-day"
    if "PerMinute" in qid:
        return "per-minute"
    return "unknown"


def park(model: str, err: Exception) -> None:
    kind = quota_kind(err)
    # A used-up daily limit won't come back in a minute: stop wasting requests on it.
    wait = 3600.0 if kind == "per-day" else retry_delay_s(err)
    _parked_until[model] = time.monotonic() + wait
    log.warning("Gemini quota hit on %s (%s limit); switching model for the next %.0fs",
                model, kind, wait)
    if kind == "unknown":
        log.warning("429 details: %s", str(err)[:600])


def make_config(types, **kwargs):
    """GenerateContentConfig with minimal thinking and function calling off."""
    if _thinking_supported and settings.gemini_thinking != "off":
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=settings.gemini_thinking.upper())
    kwargs.setdefault("max_output_tokens", 2048)
    kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    return types.GenerateContentConfig(**kwargs)


def thinking_unsupported(err: Exception) -> bool:
    """If a model rejects thinking_level, remember that and let the caller retry once."""
    global _thinking_supported
    msg = str(err).lower()
    if _thinking_supported and "thinking" in msg and ("400" in msg or "invalid" in msg):
        _thinking_supported = False
        log.warning("Model rejected thinking_level; retrying without it.")
        return True
    return False


def finish_reason(resp) -> str:
    try:
        return str(resp.candidates[0].finish_reason)
    except (AttributeError, IndexError, TypeError):
        return "unknown"
