"""Multi-provider LLM client with automatic key rotation and failover.

Priority order (as configured for this project):
    1. Gemini keys   - tried first, every available Gemini key is rotated through
    2. AWS Bedrock   - fallback when all Gemini keys are exhausted / rate limited
    3. Groq keys     - last-resort fallback

Design goals:
    * Catch rate-limit (HTTP 429) and quota errors and immediately advance to the
      next key, then the next provider - never crash, never hang.
    * Log which provider is *currently active* so it is visible in the console.
    * NEVER log API keys - only provider names are ever emitted.
    * All keys are read from environment variables; nothing is hardcoded.

Only the Python standard library is used (urllib), matching the rest of the repo.
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger("llm")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Network timeout (seconds) for a single provider attempt.
REQUEST_TIMEOUT = 60

# HTTP status codes that should trigger rotation to the next key / provider.
ROTATE_STATUS = {429, 500, 502, 503, 529}


class ProviderError(RuntimeError):
    """Raised when every provider/key has been exhausted."""


class _RateLimited(Exception):
    """Internal: signals that the current key was rate limited / quota exceeded."""


def _collect_keys(*names: str) -> list[str]:
    """Collect keys from a set of env var conventions, de-duplicated, order-preserving.

    Supports, for a base name like ``GEMINI_API_KEY``:
        GEMINI_API_KEY            -> single key
        GEMINI_API_KEYS           -> comma/space separated list
        GEMINI_API_KEY_1 .. _N    -> numbered keys
    """
    keys: list[str] = []
    for name in names:
        single = os.environ.get(name, "").strip().strip("\"'")
        if single:
            keys.append(single)
        plural = os.environ.get(name + "S", "")
        for part in plural.replace(",", " ").split():
            part = part.strip().strip("\"'")
            if part:
                keys.append(part)
        i = 1
        while True:
            val = os.environ.get(f"{name}_{i}", "").strip().strip("\"'")
            if not val:
                break
            keys.append(val)
            i += 1
    # de-dupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _post(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON and return the parsed JSON response. Raises urllib HTTPError."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _is_rate_limit(err: urllib.error.HTTPError, body: str) -> bool:
    if err.code in ROTATE_STATUS:
        return True
    lowered = body.lower()
    return any(s in lowered for s in ("rate limit", "quota", "resource_exhausted", "too many requests"))


# --------------------------------------------------------------------------- #
# Message formatting helpers
# --------------------------------------------------------------------------- #

def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system_text, non_system_messages)."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), convo


# --------------------------------------------------------------------------- #
# Provider adapters - each returns assistant text or raises _RateLimited /
# urllib.error.HTTPError / Exception.
# --------------------------------------------------------------------------- #

def _call_gemini(key: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    system, convo = _split_system(messages)
    contents = []
    for m in convo:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload: dict = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={key}"
    )
    resp = _post(url, payload, {"Content-Type": "application/json"})
    return resp["candidates"][0]["content"]["parts"][0]["text"]


def _call_bedrock(key: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    region = os.environ.get("AWS_BEDROCK_REGION", "us-east-1")
    model = os.environ.get("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")
    system, convo = _split_system(messages)
    payload: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": m["role"], "content": m["content"]} for m in convo],
    }
    if system:
        payload["system"] = system
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke"
    # Bedrock API keys are passed as a bearer token.
    resp = _post(url, payload, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    return resp["content"][0]["text"]


def _call_groq(key: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    resp = _post("https://api.groq.com/openai/v1/chat/completions", payload, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    return resp["choices"][0]["message"]["content"]


# Provider registry in priority order. Bedrock holds a single key.
def _provider_chain() -> list[tuple[str, callable, list[str]]]:
    return [
        ("gemini", _call_gemini, _collect_keys("GEMINI_API_KEY")),
        ("bedrock", _call_bedrock, _collect_keys("AWS_BEDROCK_API_KEY")),
        ("groq", _call_groq, _collect_keys("GROQ_API_KEY")),
    ]


def available_providers() -> list[str]:
    """Names of providers that have at least one key configured."""
    return [name for name, _fn, keys in _provider_chain() if keys]


def chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Send a chat ``messages`` array through the provider chain with rotation.

    Returns the assistant's text. Raises :class:`ProviderError` if all providers
    and keys are exhausted.
    """
    chain = _provider_chain()
    if not any(keys for _n, _f, keys in chain):
        raise ProviderError(
            "No provider keys found. Set at least one of GEMINI_API_KEY, "
            "AWS_BEDROCK_API_KEY, or GROQ_API_KEY (see .env.example)."
        )

    last_error: Exception | None = None
    for name, fn, keys in chain:
        if not keys:
            logger.info("provider '%s' has no keys configured - skipping", name)
            continue
        for idx, key in enumerate(keys, 1):
            logger.info("active provider: %s (key %d/%d)", name, idx, len(keys))
            try:
                return fn(key, messages, max_tokens, temperature)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:
                    body = ""
                if _is_rate_limit(e, body):
                    logger.warning(
                        "provider '%s' key %d/%d rate limited/quota (HTTP %s) - rotating",
                        name, idx, len(keys), e.code,
                    )
                    last_error = RuntimeError(f"{name} HTTP {e.code}")
                    continue
                # Non-rate-limit HTTP error: try next key, then next provider.
                logger.warning("provider '%s' HTTP %s - rotating", name, e.code)
                last_error = RuntimeError(f"{name} HTTP {e.code}: {body[:200]}")
                continue
            except urllib.error.URLError as e:
                logger.warning("provider '%s' network error (%s) - rotating", name, e.reason)
                last_error = e
                continue
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning("provider '%s' returned an unexpected payload - rotating", name)
                last_error = e
                continue
        logger.info("provider '%s' exhausted - falling back to next provider", name)

    raise ProviderError(f"All providers exhausted. Last error: {last_error}")
