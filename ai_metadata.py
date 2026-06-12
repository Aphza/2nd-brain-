import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import llm_providers
from agent_memory import (
    load_memory,
    is_processed,
    mark_processed,
    record_usage,
)

# Fallback label only; the real per-note source is the provider/model that
# actually answered (e.g. "gemini/gemini-2.5-flash"), threaded through at runtime.
DEFAULT_SOURCE = "gemini->bedrock->groq"

DEFAULT_BATCH_SIZE = 5
DEFAULT_TOKENS_PER_FILE = 150
DEFAULT_PROMPT_CHARS = 250
DEFAULT_SLEEP = 0.5

SYSTEM_PROMPT = (
    "You generate concise metadata for personal knowledge-base notes. "
    "Always respond with ONLY the requested JSON - no prose, no code fences."
)


def clean(text):
    return " ".join(text.replace("\r", "").split())


def parse_frontmatter(text):
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            raw = text[4:end]
            body = text[end + 5:].lstrip("\n")
            data = {}
            key = None
            for line in raw.splitlines():
                if not line.strip():
                    continue
                if line.lstrip().startswith("- "):
                    if key:
                        data.setdefault(key, []).append(line.strip()[2:])
                    continue
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip().strip('"')
                if v == "":
                    key = k
                    data[k] = []
                    continue
                key = None
                try:
                    data[k] = json.loads(v) if v.startswith("[") else v
                except json.JSONDecodeError:
                    data[k] = v
            return data, body
    return {}, text


def build_frontmatter(fm, source=DEFAULT_SOURCE):
    lines = ["---"]
    if fm.get("title"):
        lines.append(f"title: {json.dumps(fm['title'])}")
    if fm.get("summary"):
        lines.append(f"summary: {json.dumps(fm['summary'])}")
    if fm.get("tags"):
        lines.append("tags:")
        for t in fm["tags"]:
            lines.append(f"  - {t}")
    if fm.get("topics"):
        lines.append("topics:")
        for t in fm["topics"]:
            lines.append(f"  - {t}")
    lines.append(f"metadata_generated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"metadata_source: {source}")
    lines.append("---\n")
    return "\n".join(lines)


def load_file(p):
    return parse_frontmatter(p.read_text(encoding="utf-8"))


def save_file(p, fm, body, source=DEFAULT_SOURCE):
    p.write_text(build_frontmatter(fm, source) + body, encoding="utf-8")


def call(prompt, max_tokens):
    """Send ``prompt`` through the rotating provider chain.

    Returns the provider's :class:`llm_providers.LLMResult` (text + provider +
    model + tokens) so the caller can attribute which provider/model answered.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return llm_providers.chat(messages, max_tokens=max_tokens, temperature=0.0)


def parse_json(text):
    text = re.sub(r"^```.*?\n", "", text.strip())
    text = re.sub(r"\n```$", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def merge_tags(old, new):
    if isinstance(old, str):
        old = [old]
    seen = set()
    out = []
    for t in (old or []) + (new or []):
        t = re.sub(r"[^a-z0-9\-]", "-", str(t).lower()).strip("-")
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def process_batch(batch, mem=None):
    n = len(batch)
    prompt = [
        f'Return ONLY a JSON array of {n} objects. '
        f'Each: {{"summary":"1-2 sentences","tags":["x"],"topics":["x"]}}.'
    ]
    for i, (p, fm, body) in enumerate(batch, 1):
        prompt.append(f"\nNOTE {i}: {fm.get('title') or p.stem}\n{clean(body)[:DEFAULT_PROMPT_CHARS]}")

    result = call("\n".join(prompt), n * DEFAULT_TOKENS_PER_FILE)
    # Attribute the exact provider/model that answered (task 3).
    source = f"{result.provider}/{result.model}"
    if mem is not None:
        record_usage(mem, result.provider, result.tokens or 0)

    results = parse_json(result.text)
    if not isinstance(results, list) or len(results) != n:
        raise RuntimeError(f"Expected {n} results, got {results}")
    for (p, fm, body), meta in zip(batch, results):
        fm["title"] = fm.get("title") or p.stem
        fm["summary"] = meta.get("summary", "")
        fm["tags"] = merge_tags(fm.get("tags"), meta.get("tags"))
        fm["topics"] = meta.get("topics", [])
        save_file(p, fm, body, source)
        if mem is not None:
            mark_processed(mem, str(p.resolve()))
        print(f"updated: {p.name}  [{source}]")


def _log_failed(vault_dir: Path, batch, error: Exception) -> None:
    """Append the failed files to ``failed_files.log`` so they can be reprocessed."""
    stamp = datetime.now(timezone.utc).isoformat()
    log_path = vault_dir / "failed_files.log"
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            for p, _fm, _body in batch:
                fh.write(f"{stamp}\t{p.resolve()}\t{type(error).__name__}: {error}\n")
    except OSError as e:
        print(f"could not write failed_files.log: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    vault_dir = Path(args.dir)

    # Load persistent state before making any API call (task 5).
    mem = load_memory()
    print(f"loaded memory: {len(mem.get('processed_files', []))} files already processed "
          f"(run #{mem['session_stats']['runs']})")
    print(f"providers available: {llm_providers.available_providers() or 'NONE - check env'}")

    files = sorted(vault_dir.glob("*.md"))
    to_process = []
    skipped = 0
    for p in files:
        if is_processed(mem, str(p.resolve())):
            skipped += 1
            continue
        fm, body = load_file(p)
        if fm.get("tags"):
            continue
        to_process.append((p, fm, body))
        if args.max_files and len(to_process) >= args.max_files:
            break
    if skipped:
        print(f"skipping {skipped} file(s) already in processed_files")

    batches = [
        to_process[i:i + DEFAULT_BATCH_SIZE]
        for i in range(0, len(to_process), DEFAULT_BATCH_SIZE)
    ]
    for b in batches:
        try:
            process_batch(b, mem=mem)
        except Exception as e:
            # Never let one bad batch crash the run; log the files for reprocessing.
            print(f"batch failed ({type(e).__name__}: {e}); logging and continuing")
            _log_failed(vault_dir, b, e)
        time.sleep(DEFAULT_SLEEP)

    stats = mem["session_stats"]
    print(f"done. providers used: {stats['providers']} total_tokens={stats['total_tokens']}")


if __name__ == "__main__":
    main()
