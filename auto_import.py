import argparse
import json
from pathlib import Path
import shutil
from typing import Iterable

from parser import parse_conversations, write_markdown_files

ASSET_EXTENSIONS = {
    ".gif", ".md", ".wav", ".pptx", ".mov", ".jpg", ".docx", ".png", ".txt",
    ".jpeg", ".m4a", ".mp3", ".svg", ".webm", ".xlsx", ".flac", ".mp4", ".pdf",
}


def find_conversation_files(source: Path) -> list[Path]:
    files: list[Path] = []
    exact = source / "conversations.json"
    if exact.exists():
        files.append(exact)
    conv_dir = source / "conversations"
    if conv_dir.is_dir():
        for p in sorted(conv_dir.glob("*.json")):
            files.append(p)
    for p in sorted(source.glob("*.json")):
        if p.name in {"memories.json", "projects.json"}:
            continue
        if p.name in {"conversations.json"}:
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if isinstance(data, list) and data:
            if isinstance(data[0], dict) and "chat_messages" in data[0]:
                files.append(p)
    return files


def process_file(path: Path, output_dir: Path, dry_run: bool) -> None:
    print(f"Processing: {path}")
    if dry_run:
        print(f"  would parse {path} and write markdown to {output_dir}")
        return
    conversations = parse_conversations(path)
    if not conversations:
        print(f"  no conversations found in {path}")
        return
    write_markdown_files(conversations, output_dir)


def find_assets(source: Path) -> list[Path]:
    assets: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ASSET_EXTENSIONS:
            continue
        assets.append(path)
    return assets


def build_asset_note(output_dir: Path, asset_paths: list[Path], source: Path) -> None:
    note_path = output_dir / "attachments-index.md"
    lines = [
        "---",
        "title: Attachments Index",
        "tags:",
        "  - assets",
        "  - attachments",
        "---",
        "",
        "# Attachments Index",
        "",
        "Imported assets from the source export.",
        "",
        "## Files",
        "",
    ]
    for asset_path in asset_paths:
        rel_path = asset_path.relative_to(source)
        link_path = f"assets/{rel_path.as_posix()}"
        if asset_path.suffix.lower() in {
            ".gif", ".wav", ".mov", ".jpg", ".jpeg", ".png",
            ".mp3", ".m4a", ".svg", ".webm", ".flac", ".mp4",
        }:
            lines.append(f"- ![[{link_path}]]")
        else:
            lines.append(f"- [[{link_path}]]")
    lines.append("")
    note_path.write_text("\n".join(lines), encoding="utf-8")


def copy_assets(source: Path, output_dir: Path, dry_run: bool) -> None:
    asset_paths = find_assets(source)
    if not asset_paths:
        return
    asset_dir = output_dir / "assets"
    if dry_run:
        print(f"  would copy {len(asset_paths)} assets to {asset_dir}")
        return
    for path in asset_paths:
        rel_path = path.relative_to(source)
        target = asset_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        print(f"  copied asset: {rel_path}")
    build_asset_note(output_dir, asset_paths, source)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_to_processed(path: Path, processed_dir: Path, dry_run: bool) -> None:
    ensure_dir(processed_dir)
    target = processed_dir / path.name
    if dry_run:
        print(f"  would move {path} to {target}")
        return
    print(f"  moving {path.name} -> {target}")
    shutil.move(str(path), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Claude JSON exports into Obsidian vault as markdown notes."
    )
    parser.add_argument(
        "source", nargs="?", default="~/Downloads/Claude CONVOS",
        help="Source folder with Claude export JSON",
    )
    parser.add_argument(
        "--vault", default="~/Library/Mobile Documents/com~apple~CloudDocs/Brain",
        help="Obsidian vault folder",
    )
    parser.add_argument(
        "--output", default="AI_Processed",
        help="Subfolder inside vault for imported notes",
    )
    parser.add_argument(
        "--move-processed", action="store_true",
        help="Move processed JSON into a processed folder",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without writing files",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    vault = Path(args.vault).expanduser().resolve()
    output_dir = vault / args.output
    processed_dir = source / "processed"

    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")
    if not vault.exists() or not vault.is_dir():
        raise SystemExit(f"Vault folder not found: {vault}")

    ensure_dir(output_dir)
    copy_assets(source, output_dir, args.dry_run)

    files = find_conversation_files(source)
    if not files:
        raise SystemExit(f"No conversation JSON files found in {source}")

    for path in files:
        process_file(path, output_dir, args.dry_run)
        if not args.move_processed:
            continue
        if args.dry_run:
            continue
        move_to_processed(path, processed_dir, args.dry_run)

    print(f"Done. Output folder: {output_dir}")


if __name__ == "__main__":
    main()
