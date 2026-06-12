# Repair Summary

Autonomous repair pass on branch `claude/hyperion-osin-audit-refactor-fh70qs`.
All seven requested tasks are complete. Every module byte-compiles on CPython
3.11 and 3.14 and passes the offline test suite described below.

## What was changed

### 1. Bedrock auth — proper SigV4 (`llm_providers.py`)
- Replaced the bearer-only Bedrock adapter with **SigV4 request signing**:
  - **boto3** is used if importable (it handles SigV4 and the full AWS credential
    chain). boto3 is optional — imported lazily.
  - If boto3 is absent, a **manual HMAC-SHA256 SigV4 signer** (`_sigv4_headers`)
    built on the standard library is used. The model id is percent-encoded in
    both the request URI and the canonical URI (it contains `:`).
- **Env contract:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_SESSION_TOKEN` (optional), `AWS_BEDROCK_REGION` (default `us-east-1`),
  `AWS_BEDROCK_MODEL` (default `us.anthropic.claude-haiku-4-5-20251001-v1:0`).
- **Bearer fallback:** if `AWS_BEDROCK_API_KEY` is also present, SigV4 is tried
  first and bearer is used only if SigV4 fails. If bearer is the only credential,
  it is used directly.

### 2. Default models validated (`llm_providers.py`)
Checked against current provider docs (June 2026):
| Provider | Old default | New default | Why |
|---|---|---|---|
| Gemini | `gemini-1.5-flash` | **`gemini-2.5-flash`** | The 1.5 line is retired; 2.5-flash is the current broadly-available fast model on the `v1beta` endpoint. |
| Bedrock | `anthropic.claude-3-5-haiku-20241022-v1:0` | **`us.anthropic.claude-haiku-4-5-20251001-v1:0`** | Matches the project's intended Haiku 4.5; on-demand requires the cross-Region inference-profile id (`us.` prefix + `-v1:0` suffix, which was missing). |
| Groq | `llama-3.3-70b-versatile` | **unchanged** | Confirmed active and recommended (not deprecated). |
All three remain overridable via `GEMINI_MODEL` / `AWS_BEDROCK_MODEL` / `GROQ_MODEL`.

### 3. `metadata_source` now reflects the real provider (`llm_providers.py`, `ai_metadata.py`)
- `chat()` now returns an `LLMResult(text, provider, model, tokens)` instead of a
  bare string.
- `ai_metadata.process_batch()` stamps each note with the provider/model that
  actually answered, e.g. `metadata_source: gemini/gemini-2.5-flash` or
  `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`.
- `build_frontmatter(fm, source)` / `save_file(p, fm, body, source)` take the
  source through at runtime.

### 4. Hardened per-batch error handling (`ai_metadata.py`)
- The per-batch `try/except` in `main()` is verified to catch and log without
  crashing the run.
- On failure, every file in the batch is appended to **`failed_files.log`** in
  the vault directory, with an ISO timestamp, absolute path, and the error, so
  the user can reprocess them.

### 5. Memory redesigned for a batch job (`agent_memory.py`)
- Dropped the chat-style `messages`/history replay (wasteful, context-overflow
  risk). The file now stores only:
  - `processed_files` — absolute paths already successfully enriched.
  - `session_stats` — per-provider `{calls, tokens}` and `total_tokens`, plus a
    `runs` counter.
- On startup, files already in `processed_files` are **skipped**.
- Same filename `agent_memory.json`; the loader migrates the old v1 format.

### 6. Pull request
Opened from `claude/hyperion-osin-audit-refactor-fh70qs` into `main`
(see PR link in the run output / chat).

### 7. This summary.

## Assumptions made (documented in code comments)
- **Gemini default → `gemini-2.5-flash`.** A Gemini 3.x flash exists by mid-2026
  but its exact API id could not be verified, so I chose the most clearly
  documented, broadly available fast model and left it overridable.
- **Bedrock model id** assumes on-demand cross-Region inference (`us.` profile).
  If you are pinned to a single Region/provisioned throughput, override
  `AWS_BEDROCK_MODEL`.
- **SigV4 fallback order**: SigV4 (boto3 → manual) → bearer, per the task.
- `processed_files` is keyed by **absolute resolved path** so moving the vault
  invalidates the cache (intentional — safer than matching bare filenames).

## Needs human eyes
- **Live credentials**: no real Gemini/Bedrock/Groq/AWS calls were made — all
  network paths were exercised with mocks. Validate against real keys before
  release. In particular confirm your AWS principal has
  `bedrock:InvokeModel` on the inference-profile ARN.
- **Revoke the AWS Bedrock key** that was pasted into the chat earlier — treat
  it as compromised.
- **Manual SigV4 signer**: verified structurally and via a mocked request; run
  one real Bedrock call to confirm the signature is accepted end-to-end (or
  install `boto3`, which is the preferred path).
- **boto3 not installed** in this environment, so the manual signer is the
  active path here. `pip install boto3` to use the AWS SDK path.

## Environment variables required to run
```
# Gemini (priority 1) — at least one:
GEMINI_API_KEY=...            # or GEMINI_API_KEYS=a,b  or GEMINI_API_KEY_1=...
# GEMINI_MODEL=gemini-2.5-flash

# AWS Bedrock (priority 2) — SigV4 preferred:
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
# AWS_SESSION_TOKEN=...        # temporary creds only
# AWS_BEDROCK_REGION=us-east-1
# AWS_BEDROCK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
# AWS_BEDROCK_API_KEY=...      # optional bearer fallback

# Groq (priority 3):
GROQ_API_KEY=...               # or GROQ_API_KEYS=a,b
# GROQ_MODEL=llama-3.3-70b-versatile
```
Run: `python ai_metadata.py /path/to/vault [--max-files N]`

## Test coverage (offline, mocked network)
- SigV4 signer structure (with/without session token); manual Bedrock path with
  model-id percent-encoding; SigV4→bearer fallback.
- `chat()` returns provider/model/tokens; Gemini→Bedrock→Groq rotation on 429.
- Memory v1→v2 migration; `processed_files` skip; `session_stats` accounting.
- `ai_metadata` end-to-end: attribution written to frontmatter, processed-skip on
  re-run, and `failed_files.log` written (no crash) when all providers fail.
