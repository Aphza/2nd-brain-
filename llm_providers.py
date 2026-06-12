"""Multi-provider LLM client with automatic key rotation and failover.

Priority order (as configured for this project):
    1. Gemini keys   - tried first, every available Gemini key is rotated through
    2. AWS Bedrock   - fallback when Gemini keys are exhausted / rate limited
    3. Groq keys     - last-resort fallback

Design goals:
    * Catch rate-limit (HTTP 429) and quota errors and immediately advance to the
      next key, then the next provider - never crash, never hang.
    * Log which provider is *currently active* so it is visible in the console.
    * NEVER log API keys - only provider names are ever emitted.
    * All credentials are read from environment variables; nothing is hardcoded.

``chat()`` returns an :class:`LLMResult` (text + provider + model + tokens) so the
caller can attribute exactly which provider/model answered.

AWS Bedrock authentication (task 1):
    * SigV4 request signing is used by default - boto3 if available (it handles
      SigV4 and the full credential chain), otherwise a manual HMAC-SHA256 signer
      built on the standard library.
    * Env contract: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
      AWS_SESSION_TOKEN (optional), AWS_BEDROCK_REGION (default us-east-1),
      AWS_BEDROCK_MODEL (default us.anthropic.claude-haiku-4-5-20251001-v1:0).
    * If a bearer key (AWS_BEDROCK_API_KEY) is *also* present, SigV4 is tried
      first and bearer auth is used only as a fallback.

Only the Python standard library is required; boto3 is used opportunistically.
"""

import hashlib
import hmac
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import namedtuple
from datetime import datetime, timezone

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

# Result of a successful provider call. ``tokens`` may be None if the provider
# did not report usage.
LLMResult = namedtuple("LLMResult", ["text", "provider", "model", "tokens"])


class ProviderError(RuntimeError):
    """Raised when every provider/key has been exhausted."""


# --------------------------------------------------------------------------- #
# Default models (validated against provider docs, June 2026 - see comments).
# Each is overridable via the matching *_MODEL env var.
# --------------------------------------------------------------------------- #

def _gemini_model() -> str:
    # gemini-1.5-flash (the original default) is retired; the 1.5 line is gone.
    # gemini-2.5-flash is the current broadly-available fast model on the
    # generativelanguage v1beta endpoint as of June 2026.
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _bedrock_model() -> str:
    # Claude Haiku 4.5 on Bedrock is only served on-demand via a cross-Region
    # inference profile, whose id carries the "us." prefix AND the "-v1:0"
    # suffix (the suffix was missing from the original default).
    return os.environ.get("AWS_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")


def _bedrock_region() -> str:
    return os.environ.get("AWS_BEDROCK_REGION", "us-east-1")


def _groq_model() -> str:
    # llama-3.3-70b-versatile is active and recommended (not deprecated) as of
    # June 2026 - it is itself the replacement for the retired llama3-*-8192 ids.
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


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
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _post(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON and return the parsed JSON response. Raises urllib HTTPError."""
    return _post_bytes(url, json.dumps(payload).encode("utf-8"), headers)


def _post_bytes(url: str, body: bytes, headers: dict) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _is_rate_limit(err: urllib.error.HTTPError, body: str) -> bool:
    if err.code in ROTATE_STATUS:
        return True
    lowered = body.lower()
    return any(s in lowered for s in ("rate limit", "quota", "resource_exhausted", "too many requests"))


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system_text, non_system_messages)."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), convo


# --------------------------------------------------------------------------- #
# Provider adapters - each returns (text, tokens). ``tokens`` is the total token
# count reported by the provider, or None.
# --------------------------------------------------------------------------- #

def _call_gemini(key: str, model: str, messages: list[dict], max_tokens: int, temperature: float):
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
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    tokens = resp.get("usageMetadata", {}).get("totalTokenCount")
    return text, tokens


def _call_groq(key: str, model: str, messages: list[dict], max_tokens: int, temperature: float):
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
    text = resp["choices"][0]["message"]["content"]
    tokens = resp.get("usage", {}).get("total_tokens")
    return text, tokens


def _anthropic_payload(messages: list[dict], max_tokens: int, temperature: float) -> dict:
    system, convo = _split_system(messages)
    payload: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": m["role"], "content": m["content"]} for m in convo],
    }
    if system:
        payload["system"] = system
    return payload


def _parse_anthropic(data: dict):
    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    tokens = (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None
    return text, tokens


def _sigv4_headers(method, service, region, host, path, body, access_key, secret_key, session_token=None):
    """Build AWS SigV4 signed headers for a request (standard-library only)."""
    now = datetime.now(timezone.utc)
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()

    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amzdate}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    if session_token:
        # Header list must stay alphabetically sorted for the canonical request.
        canonical_headers += f"x-amz-security-token:{session_token}\n"
        signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"

    canonical_request = "\n".join(
        [method, path, "", canonical_headers, signed_headers, payload_hash]
    )
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        algorithm,
        amzdate,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    def _sign(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), datestamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Amz-Date": amzdate,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": (
            f"{algorithm} Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }
    if session_token:
        headers["X-Amz-Security-Token"] = session_token
    return headers


def _bedrock_sigv4_boto3(model, region, body: bytes):
    """Invoke Bedrock via boto3 (handles SigV4 + the full AWS credential chain)."""
    import boto3  # imported lazily so the dependency is optional
    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.invoke_model(
        modelId=model, body=body, contentType="application/json", accept="application/json"
    )
    return json.loads(resp["body"].read())


def _bedrock_sigv4_manual(model, region, body: bytes):
    """Invoke Bedrock with a hand-rolled SigV4 signature (no boto3)."""
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    session_token = os.environ.get("AWS_SESSION_TOKEN") or None
    host = f"bedrock-runtime.{region}.amazonaws.com"
    # The model id contains ':' (e.g. ...-v1:0) so it must be percent-encoded in
    # both the request URI and the SigV4 canonical URI.
    path = f"/model/{urllib.parse.quote(model, safe='')}/invoke"
    headers = _sigv4_headers(
        "POST", "bedrock", region, host, path, body, access_key, secret_key, session_token
    )
    return _post_bytes(f"https://{host}{path}", body, headers)


def _bedrock_bearer(model, region, body: bytes):
    """Invoke Bedrock with a bearer API key (AWS_BEDROCK_API_KEY)."""
    key = os.environ["AWS_BEDROCK_API_KEY"].strip().strip("\"'")
    host = f"bedrock-runtime.{region}.amazonaws.com"
    path = f"/model/{urllib.parse.quote(model, safe='')}/invoke"
    return _post_bytes(f"https://{host}{path}", body, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })


def _bedrock_configured() -> bool:
    has_sigv4 = bool(os.environ.get("AWS_ACCESS_KEY_ID")) and bool(os.environ.get("AWS_SECRET_ACCESS_KEY"))
    has_bearer = bool(os.environ.get("AWS_BEDROCK_API_KEY"))
    return has_sigv4 or has_bearer


def _call_bedrock(model: str, region: str, messages: list[dict], max_tokens: int, temperature: float):
    body = json.dumps(_anthropic_payload(messages, max_tokens, temperature)).encode("utf-8")
    has_sigv4 = bool(os.environ.get("AWS_ACCESS_KEY_ID")) and bool(os.environ.get("AWS_SECRET_ACCESS_KEY"))
    has_bearer = bool(os.environ.get("AWS_BEDROCK_API_KEY"))

    if has_sigv4:
        try:
            try:
                data = _bedrock_sigv4_boto3(model, region, body)
                logger.info("bedrock auth: SigV4 (boto3)")
            except ImportError:
                data = _bedrock_sigv4_manual(model, region, body)
                logger.info("bedrock auth: SigV4 (manual HMAC)")
            return _parse_anthropic(data)
        except Exception as e:
            # SigV4 failed. Fall back to bearer only if it is configured;
            # otherwise re-raise so the caller can rotate to the next provider.
            if not has_bearer:
                raise
            logger.warning("bedrock SigV4 failed (%s) - falling back to bearer key", type(e).__name__)

    # Bearer path (either chosen directly or as SigV4 fallback).
    data = _bedrock_bearer(model, region, body)
    logger.info("bedrock auth: bearer key")
    return _parse_anthropic(data)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _plan(messages, max_tokens, temperature):
    """Build the ordered execution plan: [(provider, model, [attempt_fns])]."""
    plan = []
    gkeys = _collect_keys("GEMINI_API_KEY")
    if gkeys:
        gmodel = _gemini_model()
        plan.append(("gemini", gmodel, [
            (lambda k=k: _call_gemini(k, gmodel, messages, max_tokens, temperature)) for k in gkeys
        ]))
    if _bedrock_configured():
        bmodel = _bedrock_model()
        bregion = _bedrock_region()
        plan.append(("bedrock", bmodel, [
            lambda: _call_bedrock(bmodel, bregion, messages, max_tokens, temperature)
        ]))
    qkeys = _collect_keys("GROQ_API_KEY")
    if qkeys:
        qmodel = _groq_model()
        plan.append(("groq", qmodel, [
            (lambda k=k: _call_groq(k, qmodel, messages, max_tokens, temperature)) for k in qkeys
        ]))
    return plan


def available_providers() -> list[str]:
    """Names of providers that have credentials configured, in priority order."""
    names = []
    if _collect_keys("GEMINI_API_KEY"):
        names.append("gemini")
    if _bedrock_configured():
        names.append("bedrock")
    if _collect_keys("GROQ_API_KEY"):
        names.append("groq")
    return names


def chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> LLMResult:
    """Send a chat ``messages`` array through the provider chain with rotation.

    Returns an :class:`LLMResult` (text, provider, model, tokens). Raises
    :class:`ProviderError` if all providers and keys are exhausted.
    """
    plan = _plan(messages, max_tokens, temperature)
    if not plan:
        raise ProviderError(
            "No provider credentials found. Set at least one of GEMINI_API_KEY, "
            "AWS credentials / AWS_BEDROCK_API_KEY, or GROQ_API_KEY (see .env.example)."
        )

    last_error: Exception | None = None
    for name, model, attempts in plan:
        n = len(attempts)
        for idx, attempt in enumerate(attempts, 1):
            logger.info("active provider: %s/%s (key %d/%d)", name, model, idx, n)
            try:
                text, tokens = attempt()
                return LLMResult(text, name, model, tokens)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:
                    body = ""
                if _is_rate_limit(e, body):
                    logger.warning("provider '%s' key %d/%d rate limited (HTTP %s) - rotating",
                                   name, idx, n, e.code)
                else:
                    logger.warning("provider '%s' HTTP %s - rotating", name, e.code)
                last_error = RuntimeError(f"{name} HTTP {e.code}")
                continue
            except urllib.error.URLError as e:
                logger.warning("provider '%s' network error (%s) - rotating", name, e.reason)
                last_error = e
                continue
            except Exception as e:
                # Includes botocore throttling/ClientError and malformed payloads.
                logger.warning("provider '%s' attempt failed (%s) - rotating", name, type(e).__name__)
                last_error = e
                continue
        logger.info("provider '%s' exhausted - falling back to next provider", name)

    raise ProviderError(f"All providers exhausted. Last error: {last_error}")
