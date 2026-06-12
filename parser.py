import argparse
import json
from pathlib import Path
import re
from typing import Optional


def clean_text(text: str) -> str:
    return text.replace("\r", "").strip()


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def normalize_tag(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", text.lower()).strip("-")


def extract_attachments(conversation: dict) -> list[dict]:
    attachments: list[dict] = []
    for key in ("attachments", "media", "files"):
        items = conversation.get(key)
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                attachments.append({"name": item, "type": "file", "url": ""})
            elif isinstance(item, dict):
                name = item.get("name") or item.get("filename") or item.get("url")
                if not name:
                    continue
                attachments.append({
                    "name": str(name),
                    "type": item.get("type") or item.get("media_type") or "file",
                    "url": item.get("url") or item.get("path") or "",
                })
    return attachments


def extract_topic_tags(title: str, limit: int = 3) -> list[str]:
    stopwords = {
        "that", "should", "the", "with", "about", "have", "their", "then",
        "were", "them", "could", "when", "because", "your", "just", "will",
        "would", "what", "which", "more", "this", "for", "from", "and",
    }
    words = re.split(r"[^A-Za-z0-9]+", title.lower())
    candidates = [w for w in words if len(w) > 3 and w not in stopwords]
    tags: list[str] = []
    for word in candidates:
        tag = normalize_tag(word)
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def build_frontmatter(title: str, created_at: Optional[str], summary: str, attachments: list[dict]) -> list[str]:
    lines = ["---"]
    lines.append(f"title: {yaml_string(title)}")
    if created_at:
        lines.append(f"date: {yaml_string(created_at)}")
    if summary:
        lines.append(f"summary: {yaml_string(summary)}")
    if attachments:
        lines.append("attachments:")
        for attachment in attachments:
            lines.append(f"  - {yaml_string(attachment['name'])}")
    tags = ["ai", "claude", "secondbrain", "conversation"]
    tags.extend(extract_topic_tags(title))
    tags = [t for i, t in enumerate(tags) if t and t not in tags[:i]]
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")
    return lines


def build_markdown(conversation: dict, index: int) -> str:
    title = conversation.get("name") or conversation.get("summary") or f"Conversation {index + 1}"
    title = clean_text(title)
    if not title:
        title = f"Conversation {index + 1}"
    created_at = conversation.get("created_at")
    summary = clean_text(conversation.get("summary", ""))
    attachments = extract_attachments(conversation)
    lines = build_frontmatter(title, created_at, summary, attachments)
    lines.extend([f"# {title}", ""])
    if summary:
        lines.extend(["## Summary", "", summary, ""])
    lines.extend(["## Conversation", ""])
    for message in conversation.get("chat_messages", []):
        sender = str(message.get("sender", "")).lower()
        if sender in {"user", "human"}:
            speaker = "User"
        elif sender in {"bot", "claude", "ai", "assistant"}:
            speaker = "Claude"
        else:
            speaker = sender.title() or "Message"
        text = message.get("text") or ""
        if not text and message.get("content"):
            content_items = []
            for item in message["content"]:
                if item.get("type") == "text":
                    content_items.append(item.get("text", ""))
            text = "\n".join(content_items)
        text = clean_text(str(text))
        if not text:
            continue
        for paragraph in text.splitlines():
            if paragraph.strip():
                lines.append(f"{speaker}: {paragraph.strip()}")
            else:
                lines.append("")
        lines.append("")
    if attachments:
        lines.extend(["## Attachments", ""])
        for attachment in attachments:
            if attachment.get("url"):
                lines.append(f"- [{attachment['name']}]({attachment['url']})")
            else:
                lines.append(f"- {attachment['name']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_conversations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data.get("conversations") or data.get("items") or []
    return data


def safe_filename(name: str, default: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    cleaned = cleaned[:150].rstrip(".")
    return cleaned or default


def write_markdown_files(conversations: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, conversation in enumerate(conversations):
        title = conversation.get("name") or conversation.get("summary") or ""
        filename = safe_filename(title, f"conversation-{index + 1}")
        target = out_dir / f"{filename}.md"
        target.write_text(build_markdown(conversation, index), encoding="utf-8")
        print(f"wrote: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Claude conversation JSON into Obsidian markdown notes."
    )
    parser.add_argument("input", help="Path to conversations.json")
    parser.add_argument("--out", "-o", default="output", help="Output directory for markdown files")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    conversations = parse_conversations(input_path)
    if not conversations:
        raise SystemExit("No conversations found in the JSON file.")

    write_markdown_files(conversations, Path(args.out))


if __name__ == "__main__":
    main()
