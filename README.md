# 2nd-brain-

My 2nd brain — a small toolkit for turning exported Claude/ChatGPT conversations
into an interlinked Obsidian-style Markdown vault, and enriching the notes with
AI-generated metadata.

## Pipeline

```
conversations.json
      │  parser.py / auto_import.py      (export → Markdown notes + assets)
      ▼
  Markdown vault
      │  ai_metadata.py                  (AI: summary / tags / topics frontmatter)
      ▼
  enriched notes
      │  link_extractor.py / note_graph.py   (wikilinks, backlinks, relationships)
      ▼
  vault_stress_test.py                   (validation)
```

## Modules

| Module | Purpose |
|--------|---------|
| `parser.py` | Convert `conversations.json` into Markdown notes with YAML frontmatter. |
| `auto_import.py` | Import conversation JSON + assets into a vault folder; optional move-to-processed. |
| `link_extractor.py` | Extract `[[wikilinks]]`, compute `linked_notes` / `backlinks`. |
| `note_graph.py` | Score note relationships and inject a Connections section. |
| `vault_stress_test.py` | Validate frontmatter and that wikilinks resolve. |
| `ai_metadata.py` | Batch-generate `summary`/`tags`/`topics` via the multi-provider LLM client. |
| `llm_providers.py` | Provider client with key rotation: **Gemini → Bedrock → Groq**. |
| `agent_memory.py` | Persistent conversation/context memory (`agent_memory.json`). |

## API keys & provider rotation

Keys are loaded **only** from environment variables (see `.env.example`).
`ai_metadata.py` calls `llm_providers.chat()`, which tries providers in order:

1. **Gemini** — rotates through every configured Gemini key first.
2. **AWS Bedrock** — used when all Gemini keys are exhausted or rate limited.
3. **Groq** — last-resort fallback.

Rate-limit (HTTP 429) and quota errors are caught automatically and trigger an
immediate switch to the next key, then the next provider — no crashes, no hangs.
The **active provider name** is logged to the console; **keys are never logged**.

```bash
cp .env.example .env   # fill in your keys
export $(grep -v '^#' .env | xargs)   # or use a dotenv loader
python ai_metadata.py path/to/vault --max-files 20
```

## Memory

`ai_metadata.py` loads `agent_memory.json` at the start of every run, passes the
prior conversation as a `messages` array on every call, and re-persists after
each turn. Delete the file to start fresh. It is git-ignored by default.

## Note on source

These modules were reconstructed from compiled `.pyc` bytecode. The original
`*.cpython-314.pyc` files are retained as a reference and can be removed once you
have confirmed the source behaves as expected.
