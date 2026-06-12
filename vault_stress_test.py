import argparse
import json
import re
import sys
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


def load_note(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    return parse_frontmatter(text)


def normalize_title(title: str) -> str:
    return title.strip().lower()


def extract_wikilinks(text: str) -> list[str]:
    return [m.group(1).strip() for m in WIKILINK_RE.finditer(text)]


def validate_note(path: Path, title_map: dict) -> list[str]:
    errors: list[str] = []
    fm, body = load_note(path)
    title = fm.get("title") or path.stem
    if not title:
        errors.append("missing title")
    if "soul" in fm and not fm["soul"]:
        errors.append("empty soul metadata")
    if "mind" in fm and not fm["mind"]:
        errors.append("empty mind metadata")
    if "tonality" in fm and not fm["tonality"]:
        errors.append("empty tonality metadata")
    if "atlas" in fm and not fm["atlas"]:
        errors.append("empty atlas metadata")
    outgoing = extract_wikilinks(body)
    for link in outgoing:
        if normalize_title(link) not in title_map:
            errors.append(f"broken wikilink: {link}")
    if fm.get("related_notes") and not isinstance(fm["related_notes"], list):
        errors.append("related_notes must be a list")
    if fm.get("backlinks") and not isinstance(fm["backlinks"], list):
        errors.append("backlinks must be a list")
    return errors


def run_validation(note_dir: Path) -> tuple[int, int, dict]:
    files = sorted(note_dir.glob("*.md"))
    title_map: dict = {}
    for path in files:
        fm, _ = load_note(path)
        title = fm.get("title") or path.stem
        title_map[normalize_title(title)] = path

    notes_checked = 0
    total_errors = 0
    error_summary: dict = {}

    for path in files:
        notes_checked += 1
        errors = validate_note(path, title_map)
        if not errors:
            continue
        total_errors += len(errors)
        error_summary[path.name] = errors

    return notes_checked, total_errors, error_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated dry-run validation on an Obsidian vault note folder."
    )
    parser.add_argument("directory", help="Folder containing markdown notes")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of validation cycles")
    parser.add_argument("--report-every", type=int, default=100, help="Log progress every N cycles")
    parser.add_argument("--max-errors", type=int, default=10, help="Stop early after this many distinct note errors")
    args = parser.parse_args()

    note_dir = Path(args.directory).expanduser().resolve()
    if not note_dir.is_dir():
        raise SystemExit(f"Not a directory: {note_dir}")

    all_errors: dict = {}
    for i in range(1, args.iterations + 1):
        notes_checked, total_errors, error_summary = run_validation(note_dir)
        if total_errors:
            for note, errors in error_summary.items():
                all_errors.setdefault(note, set()).update(errors)
            if len(all_errors) >= args.max_errors:
                print(f"Stopping early at iteration {i} with {len(all_errors)} notes failing.")
                break
        if i % args.report_every == 0 or total_errors:
            print(f"iteration {i}: notes={notes_checked} errors={total_errors}")
            if total_errors:
                for note, errors in sorted(error_summary.items()):
                    print(f"  {note}: {', '.join(errors)}")

    print("\nvalidation complete")
    print(f"iterations run: {i}")
    print(f"distinct notes with errors: {len(all_errors)}")
    if all_errors:
        print("sample failures:")
        for note, errors in sorted(all_errors.items())[:10]:
            print(f"  {note}: {', '.join(sorted(errors))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
