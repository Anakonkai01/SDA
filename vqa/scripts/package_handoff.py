import argparse
import shutil
from pathlib import Path


MINIMAL_ITEMS = [
    "README.md",
    "README_HANDOFF.md",
    "PROJECT_STATE.md",
    "HANDOFF_CONTEXT.md",
    "CLAUDE.md",
    "requirements.txt",
    ".env.example",
    "docs",
    "models",
    "train",
    "evaluate",
    "data_utils",
    "scripts",
    "src",
    "prompts",
    "data/processed",
]


FULL_EXCLUDES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
    ".venv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the VQA project for teammate or remote-GPU handoff.",
    )
    parser.add_argument(
        "--mode",
        choices=["minimal", "full"],
        default="full",
        help="minimal = curated subset, full = full vqa directory except junk.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to create the handoff directory.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also create a zip archive next to the output directory.",
    )
    return parser.parse_args()


def ignore_filter(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in FULL_EXCLUDES:
            ignored.add(name)
            continue
        if name.endswith(".pyc"):
            ignored.add(name)
    return ignored


def copy_item(src_root: Path, dst_root: Path, relative_path: str) -> None:
    src = src_root / relative_path
    dst = dst_root / relative_path
    if not src.exists():
        raise FileNotFoundError(f"Missing required path: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore_filter)
    else:
        shutil.copy2(src, dst)


def package_minimal(vqa_root: Path, handoff_vqa_root: Path) -> None:
    for item in MINIMAL_ITEMS:
        copy_item(vqa_root, handoff_vqa_root, item)


def package_full(vqa_root: Path, handoff_vqa_root: Path) -> None:
    shutil.copytree(vqa_root, handoff_vqa_root, ignore=ignore_filter)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    vqa_root = repo_root / "vqa"

    default_dir = (
        "handoff/vqa_full_handoff"
        if args.mode == "full"
        else "handoff/vqa_minimal_handoff"
    )
    output_dir = (repo_root / (args.output_dir or default_dir)).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handoff_vqa_root = output_dir / "vqa"
    if args.mode == "full":
        package_full(vqa_root, handoff_vqa_root)
    else:
        handoff_vqa_root.mkdir(parents=True, exist_ok=True)
        package_minimal(vqa_root, handoff_vqa_root)

    if args.archive:
        archive_base = output_dir.parent / output_dir.name
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )

    print(f"Mode: {args.mode}")
    print(f"Created handoff folder: {output_dir}")
    if args.archive:
        print(f"Created archive: {output_dir}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

