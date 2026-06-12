import argparse
import json
import re
from pathlib import Path

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


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


def extract_wikilinks(text: str) -> list[str]:
    return [match.group(1).strip() for match in WIKILINK_RE.finditer(text)]


def load_note(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    return parse_frontmatter(text)


def save_note(path: Path, frontmatter: dict, body: str) -> None:
    path.write_text(build_frontmatter(frontmatter) + body.lstrip("\n"), encoding="utf-8")


def normalize_title(title: str) -> str:
    return title.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract wikilink references and build related-note metadata."
    )
    parser.add_argument("directory", help="Folder containing markdown notes")
    parser.add_argument("--force", action="store_true", help="Overwrite related metadata if already present")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    args = parser.parse_args()

    note_dir = Path(args.directory)
    if not note_dir.is_dir():
        raise SystemExit(f"Not a directory: {note_dir}")

    notes = []
    for path in sorted(note_dir.glob("*.md")):
        fm, body = load_note(path)
        title = fm.get("title") or path.stem
        notes.append({
            "path": path,
            "title": title,
            "title_key": normalize_title(title),
            "frontmatter": fm,
            "body": body,
        })

    title_map = {note["title_key"]: note for note in notes}

    for note in notes:
        outgoing = extract_wikilinks(note["body"])
        linked = []
        for link in outgoing:
            link_key = normalize_title(link)
            if link_key not in title_map:
                continue
            if title_map[link_key]["title_key"] != note["title_key"]:
                linked.append(title_map[link_key]["title"])
        note["linked_notes"] = sorted(dict.fromkeys(linked))

    backlinks = {note["title"]: [] for note in notes}
    for note in notes:
        for target in note["linked_notes"]:
            backlinks.setdefault(target, []).append(note["title"])

    for note in notes:
        if not args.force and note["frontmatter"].get("linked_notes"):
            continue
        if not args.force and note["frontmatter"].get("backlinks"):
            continue
        note["frontmatter"]["linked_notes"] = note["linked_notes"]
        note["frontmatter"]["backlinks"] = sorted(backlinks.get(note["title"], []))

    for note in notes:
        if args.dry_run:
            print(f"{note['path'].name}: linked={note['linked_notes']} "
                  f"backlinks={note['frontmatter'].get('backlinks', [])}")
        else:
            save_note(note["path"], note["frontmatter"], note["body"])
            print(f"wrote: {note['path'].name}")


if __name__ == "__main__":
    main()
