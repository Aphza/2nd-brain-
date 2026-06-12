import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import llm_providers
from agent_memory import (
    load_memory,
    record_turn,
    mark_processed,
    build_messages,
)

# Model/label used to stamp generated notes. Actual provider is chosen at call
# time by llm_providers (Gemini -> Bedrock -> Groq).
METADATA_SOURCE = "gemini->bedrock->groq"

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


def build_frontmatter(fm):
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
    lines.append(f"metadata_source: {METADATA_SOURCE}")
    lines.append("---\n")
    return "\n".join(lines)


def load_file(p):
    return parse_frontmatter(p.read_text(encoding="utf-8"))


def save_file(p, fm, body):
    p.write_text(build_frontmatter(fm) + body, encoding="utf-8")


def call(prompt, max_tokens, mem=None):
    """Send ``prompt`` through the rotating provider chain.

    Conversation history is always passed as a ``messages`` array. When ``mem``
    is provided, prior turns are replayed and the new turn is persisted.
    """
    if mem is not None:
        messages = build_messages(mem, SYSTEM_PROMPT, prompt)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    text = llm_providers.chat(messages, max_tokens=max_tokens, temperature=0.0)
    if mem is not None:
        record_turn(mem, prompt, text)
    return text


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
    raw = call("\n".join(prompt), n * DEFAULT_TOKENS_PER_FILE, mem=mem)
    results = parse_json(raw)
    if not isinstance(results, list) or len(results) != n:
        raise RuntimeError(f"Expected {n} results, got {results}")
    for (p, fm, body), meta in zip(batch, results):
        fm["title"] = fm.get("title") or p.stem
        fm["summary"] = meta.get("summary", "")
        fm["tags"] = merge_tags(fm.get("tags"), meta.get("tags"))
        fm["topics"] = meta.get("topics", [])
        save_file(p, fm, body)
        if mem is not None:
            mark_processed(mem, p.name)
        print("updated:", p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    # Load persistent memory before making any API call (Step 4).
    mem = load_memory()
    print(f"loaded memory: {len(mem.get('history', []))} prior turns, "
          f"{len(mem.get('processed', []))} files already processed")
    print(f"providers available: {llm_providers.available_providers() or 'NONE - check env'}")

    files = sorted(Path(args.dir).glob("*.md"))
    to_process = []
    for p in files:
        fm, body = load_file(p)
        if fm.get("tags"):
            continue
        to_process.append((p, fm, body))
        if args.max_files and len(to_process) >= args.max_files:
            break

    batches = [
        to_process[i:i + DEFAULT_BATCH_SIZE]
        for i in range(0, len(to_process), DEFAULT_BATCH_SIZE)
    ]
    for b in batches:
        try:
            process_batch(b, mem=mem)
        except Exception as e:
            # Never let one bad batch crash the whole run.
            print(f"batch failed ({e}); continuing with next batch")
        time.sleep(DEFAULT_SLEEP)


if __name__ == "__main__":
    main()
