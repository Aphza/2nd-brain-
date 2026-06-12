"""Lightweight persistent state for the batch metadata tooling.

This is a *batch* job, not a chat agent, so replaying a chat-style messages
array on every call is wasteful and risks overflowing the model context. Instead
this module persists only what is useful between runs:

    * ``processed_files`` - absolute paths of notes already successfully enriched,
      so a re-run skips them instead of re-spending tokens.
    * ``session_stats``   - per-provider call counts and total tokens used.

State lives in a single ``agent_memory.json`` file, loaded once before the first
API call and re-saved (atomically) after every successful file/turn.
"""

import json
from pathlib import Path

MEMORY_PATH = Path("agent_memory.json")
MEMORY_VERSION = 2


def _fresh() -> dict:
    return {
        "version": MEMORY_VERSION,
        "processed_files": [],
        "session_stats": {"providers": {}, "total_tokens": 0, "runs": 0},
    }


def load_memory(path: Path = MEMORY_PATH) -> dict:
    """Load persisted state, migrating older formats, or return a fresh structure."""
    mem = _fresh()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Migrate the v1 chat-memory format (had "history"/"processed").
                if "processed_files" not in data and "processed" in data:
                    data["processed_files"] = data.get("processed", [])
                mem["processed_files"] = list(dict.fromkeys(data.get("processed_files", [])))
                stats = data.get("session_stats", {})
                if isinstance(stats, dict):
                    mem["session_stats"]["providers"] = dict(stats.get("providers", {}))
                    mem["session_stats"]["total_tokens"] = stats.get("total_tokens", 0)
                    mem["session_stats"]["runs"] = stats.get("runs", 0)
        except (json.JSONDecodeError, OSError):
            pass
    mem["session_stats"]["runs"] += 1
    return mem


def save_memory(mem: dict, path: Path = MEMORY_PATH) -> None:
    """Persist state atomically to avoid corrupting the file on crash."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_processed(mem: dict, path) -> bool:
    """True if ``path`` has already been successfully enriched in a prior run."""
    return str(path) in set(mem.get("processed_files", []))


def mark_processed(mem: dict, path, mem_path: Path = MEMORY_PATH) -> None:
    """Record that ``path`` was enriched and persist immediately."""
    key = str(path)
    processed = mem.setdefault("processed_files", [])
    if key not in processed:
        processed.append(key)
        save_memory(mem, mem_path)


def record_usage(mem: dict, provider: str, tokens: int = 0, mem_path: Path = MEMORY_PATH) -> None:
    """Increment per-provider call counts and total token usage, then persist."""
    stats = mem.setdefault("session_stats", {"providers": {}, "total_tokens": 0, "runs": 0})
    providers = stats.setdefault("providers", {})
    entry = providers.setdefault(provider, {"calls": 0, "tokens": 0})
    entry["calls"] += 1
    entry["tokens"] += int(tokens or 0)
    stats["total_tokens"] = stats.get("total_tokens", 0) + int(tokens or 0)
    save_memory(mem, mem_path)
