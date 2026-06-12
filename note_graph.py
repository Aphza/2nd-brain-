import argparse
import json
import re
from datetime import datetime
from pathlib import Path

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
GENERIC_TAGS = {
    "import", "ai", "secondbrain", "chat", "notes", "journal",
    "note", "conversation", "claude",
}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            raw = text[4:end]
            body = text[end + 5:].lstrip("\n")
            data: dict = {}
            key = None
            for line in raw.splitlines():
                if not line.strip():
                    continue
                if line.lstrip().startswith("- ") and key:
                    data.setdefault(key, []).append(line.strip()[2:])
                    continue
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v == "":
                    data[k] = []
                    key = k
                    continue
                key = None
                if v.startswith("[") and v.endswith("]"):
                    try:
                        data[k] = json.loads(v)
                    except json.JSONDecodeError:
                        data[k] = [item.strip().strip('"') for item in v[1:-1].split(",") if item.strip()]
                else:
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    data[k] = v
            return data, body
    return {}, text


def build_frontmatter(fm: dict) -> str:
    lines = ["---"]
    for key, value in fm.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_date(value: str):
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def classify_node_type(title: str, tags: set, topics: set) -> str:
    low = title.lower()
    if any(word in low for word in ("plan", "goal", "study", "exam", "project", "task")):
        return "action"
    if any(word in low for word in ("system", "framework", "structure", "process")):
        return "system"
    if any(word in low for word in ("why", "insight", "meaning", "learn", "understand")):
        return "insight"
    if tags & {"self", "reflection", "thought", "journal"}:
        return "reflection"
    return "idea"


def build_context_snapshot(fm: dict, title: str, body: str) -> dict:
    goal = fm.get("summary") or title
    mental_frame = "curious"
    if any(word in body.lower() for word in ("stress", "urgent", "deadline", "pressure")):
        mental_frame = "stressed"
    elif any(word in body.lower() for word in ("experiment", "test", "evaluate")):
        mental_frame = "experimental"
    elif any(word in body.lower() for word in ("plan", "goal", "schedule", "organize")):
        mental_frame = "strategic"
    env = "research"
    if any(word in title.lower() for word in ("exam", "quiz", "study", "chapter")):
        env = "learning"
    elif any(word in title.lower() for word in ("code", "project", "app", "dashboard")):
        env = "coding"
    elif any(word in title.lower() for word in ("money", "stock", "ETF", "portfolio", "investment")):
        env = "analysis"
    return {"goal": goal, "mental_frame": mental_frame, "environment": env}


def strip_connections_section(body: str) -> str:
    if "## Connections" not in body:
        return body
    return re.sub(r"## Connections\n.*$", "", body, flags=re.S).strip()


def load_note(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    return parse_frontmatter(text)


def save_note(path: Path, frontmatter: dict, body: str) -> None:
    path.write_text(build_frontmatter(frontmatter) + body.lstrip("\n"), encoding="utf-8")


def score_pair(a: dict, b: dict) -> float:
    score = 0.0
    tags_a = a.get("tags", set()) - GENERIC_TAGS
    tags_b = b.get("tags", set()) - GENERIC_TAGS
    topic_a = a.get("topics", set())
    topic_b = b.get("topics", set())
    if tags_a and tags_b:
        score += len(tags_a & tags_b) * 4
    if topic_a and topic_b:
        score += len(topic_a & topic_b) * 3
    score += min(len(a.get("title_words", set()) & b.get("title_words", set())), 2)
    score += min(len(a.get("body_words", set()) & b.get("body_words", set())), 2)
    if b["title"] in a.get("wikilinks", []) or a["title"] in b.get("wikilinks", []):
        score += 4
    return score


def relationship_type(a: dict, b: dict) -> str:
    if b["title"] in a.get("wikilinks", []):
        return "depends on"
    if a["title"] in b.get("wikilinks", []):
        return "supports"
    if a.get("tags") and b.get("tags") and len(a["tags"] & b["tags"]) >= 2:
        return "extends"
    if a.get("topics") and b.get("topics") and len(a["topics"] & b["topics"]) >= 2:
        return "supports"
    if len(a.get("body_words", set()) & b.get("body_words", set())) >= 5:
        return "influences"
    return "relates to"


def classify_pattern(note: dict, title_map: dict) -> str:
    related = note.get("related_notes", [])
    if not related:
        return "island"
    if len(related) == 1:
        return "leaf"
    mutual = sum(
        1
        for title in related
        if title in title_map and note["title"] in title_map[title].get("related_notes", [])
    )
    if len(related) == 2:
        if mutual >= 1:
            return "snake"
        return "branch"
    if len(related) >= 3:
        if mutual >= 2:
            return "orb"
        return "branch"
    return "branch"


def build_relationship_section(relationships: list, pattern: str) -> str:
    if not relationships:
        return ""
    lines = ["## Connections", "", f"**Connection pattern:** {pattern}", ""]
    grouped: dict = {}
    for title, rel in relationships:
        grouped.setdefault(rel, []).append(title)
    for rel, titles in grouped.items():
        lines.append(f"### {rel.title()}")
        lines.extend(f"- [[{title}]]" for title in titles)
        lines.append("")
    return "\n".join(lines)


def build_evolution_chain(note: dict, title_map: dict) -> list:
    chain: list = []
    current = note
    while current and current.get("origin_note"):
        chain.insert(0, current["origin_note"])
        current = title_map.get(current["origin_note"])
    if note["title"] not in chain:
        chain.append(note["title"])
    return chain


def inject_connections(body: str, relationships: list, pattern: str) -> str:
    if not relationships:
        return body
    section = build_relationship_section(relationships, pattern)
    if "## Connections" in body:
        body = re.sub(r"## Connections\n.*?(?=\n## |\Z)", section, body, flags=re.S)
        return body
    body = body.rstrip() + "\n\n" + section
    return body


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build related notes metadata from tags, topics, titles, and optional wikilinks."
    )
    parser.add_argument("directory", help="Folder containing markdown notes")
    parser.add_argument("--top", type=int, default=5, help="Number of related notes to store")
    parser.add_argument("--min-score", type=int, default=2, help="Minimum score to accept a related note")
    parser.add_argument("--force", action="store_true", help="Overwrite existing related_notes data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing files")
    args = parser.parse_args()

    note_dir = Path(args.directory)
    if not note_dir.is_dir():
        raise SystemExit(f"Not a directory: {note_dir}")

    notes = []
    for path in sorted(note_dir.glob("*.md")):
        fm, body = load_note(path)
        title = fm.get("title") or path.stem
        tags = set(t.lower() for t in fm.get("tags", []) if isinstance(t, str))
        topics = set(t.lower() for t in fm.get("topics", []) if isinstance(t, str))
        content = strip_connections_section(body)
        created_at = parse_date(fm.get("date", ""))
        if created_at is None:
            created_at = datetime.fromtimestamp(path.stat().st_mtime)
        notes.append({
            "path": path,
            "title": title,
            "frontmatter": fm,
            "body": body,
            "created_at": created_at,
            "title_norm": normalize(title),
            "title_words": set(normalize(title).split()),
            "body_words": set(normalize(content).split()),
            "tags": tags - GENERIC_TAGS,
            "topics": topics,
            "wikilinks": [m.group(1).strip() for m in WIKILINK_RE.finditer(content)],
            "node_type": classify_node_type(title, tags, topics),
            "context_snapshot": build_context_snapshot(fm, title, content),
        })

    title_map = {note["title"]: note for note in notes}

    for note in notes:
        scored = []
        for other in notes:
            if other["path"] == note["path"]:
                continue
            score = score_pair(note, other)
            if score >= args.min_score:
                rel_type = relationship_type(note, other)
                scored.append((score, other["title"], rel_type, other["created_at"]))
        scored.sort(key=lambda item: (-item[0], item[3], item[1]))

        parent = None
        child = None
        lateral = None
        shadow = []
        earlier = [item for item in scored if item[3] < note["created_at"]]
        later = [item for item in scored if item[3] > note["created_at"]]
        if earlier:
            parent = earlier[0]
        if later:
            child = later[0]
        for item in scored:
            title = item[1]
            if parent and title == parent[1]:
                continue
            if child and title == child[1]:
                continue
            if lateral is None and item[0] >= args.min_score:
                lateral = item
                continue
            if item[0] >= max(args.min_score - 1, 1):
                shadow.append(item)
            if len(shadow) >= 3:
                break

        relationships = []
        if parent:
            relationships.append((parent[1], "depends on"))
            note["origin_note"] = parent[1]
        else:
            note["origin_note"] = None
        if child:
            relationships.append((child[1], "supports"))
            note["evolved_note"] = child[1]
        else:
            note["evolved_note"] = None
        if lateral:
            relationships.append((lateral[1], "relates to"))
            note["lateral_note"] = lateral[1]
        else:
            note["lateral_note"] = None
        note["relationships"] = relationships
        note["related_notes"] = [title for title, _ in relationships]
        note["shadow_links"] = [item[1] for item in shadow if item[1] not in note["related_notes"]][:3]

    for note in notes:
        if note["wikilinks"]:
            note["frontmatter"]["wikilinks"] = sorted(dict.fromkeys(note["wikilinks"]))
        pattern = classify_pattern(note, title_map)
        note["frontmatter"]["connection_pattern"] = pattern
        note["frontmatter"]["relationships"] = [f"{title} ({rel})" for title, rel in note["relationships"]]
        note["frontmatter"]["origin_note"] = note.get("origin_note")
        note["frontmatter"]["evolved_note"] = note.get("evolved_note")
        note["frontmatter"]["lateral_note"] = note.get("lateral_note")
        note["frontmatter"]["shadow_links"] = note["shadow_links"]
        note["frontmatter"]["node_type"] = note.get("node_type")
        note["frontmatter"]["context_snapshot"] = note.get("context_snapshot")
        note["frontmatter"]["evolution_chain"] = build_evolution_chain(note, title_map)

        if not args.force and note["frontmatter"].get("related_notes"):
            note["body"] = inject_connections(note["body"], note["relationships"], pattern)
            continue
        note["frontmatter"]["related_notes"] = note["related_notes"]
        note["frontmatter"]["connections"] = note["related_notes"]
        note["body"] = inject_connections(note["body"], note["relationships"], pattern)

    for note in notes:
        if args.dry_run:
            print(f"{note['path'].name}: related={note['related_notes']}")
        else:
            save_note(note["path"], note["frontmatter"], note["body"])
            print(f"wrote: {note['path'].name}")


if __name__ == "__main__":
    main()
