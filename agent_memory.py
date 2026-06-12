"""Lightweight persistent conversation/context memory for the AI tooling.

The metadata tooling is a batch job rather than a chat agent, but the project
asked for durable memory so the agent "never forgets what it has been told or
already built". This module provides:

    * a single ``agent_memory.json`` file in the working directory,
    * loaded once at the start of a session (before any API call),
    * appended to after every turn (request + response), then re-persisted.

History is passed to the model as a ``messages`` array on every call. To keep
prompts within token limits, :func:`build_messages` only replays the most recent
``MAX_HISTORY_TURNS`` turns, while the JSON file keeps the full record.
"""

import json
from pathlib import Path

MEMORY_PATH = Path("agent_memory.json")
MEMORY_VERSION = 1

# How many recent turns to replay into the prompt (full history is still saved).
MAX_HISTORY_TURNS = 20


def load_memory(path: Path = MEMORY_PATH) -> dict:
    """Load persisted memory, or return a fresh structure if none/corrupt."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "history" in data:
                data.setdefault("processed", [])
                data.setdefault("version", MEMORY_VERSION)
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": MEMORY_VERSION, "history": [], "processed": []}


def save_memory(mem: dict, path: Path = MEMORY_PATH) -> None:
    """Persist memory atomically to avoid corrupting the file on crash."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_turn(mem: dict, user: str, assistant: str, path: Path = MEMORY_PATH) -> None:
    """Append a user/assistant turn to memory and persist immediately."""
    mem.setdefault("history", []).append({"role": "user", "content": user})
    mem["history"].append({"role": "assistant", "content": assistant})
    save_memory(mem, path)


def mark_processed(mem: dict, name: str, path: Path = MEMORY_PATH) -> None:
    """Remember that a note/file has been handled so re-runs can skip it."""
    processed = mem.setdefault("processed", [])
    if name not in processed:
        processed.append(name)
        save_memory(mem, path)


def build_messages(mem: dict, system: str, user: str) -> list[dict]:
    """Build a messages array: system + recent history + the new user turn."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    history = mem.get("history", [])
    if MAX_HISTORY_TURNS:
        history = history[-(MAX_HISTORY_TURNS * 2):]
    messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages
