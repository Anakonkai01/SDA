from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

OPENROUTER_API_KEY = ""  # set via env: export OPENROUTER_API_KEY=...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenRouter pilot labeling on the review queue, then filter results."
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenRouter API key. If omitted, use OPENROUTER_API_KEY from the environment.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process for the pilot.")
    parser.add_argument(
        "--model",
        default="google/gemma-3-27b-it:free",
        help="OpenRouter model id. Default: google/gemma-3-27b-it:free",
    )
    return parser.parse_args()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key:
        print(
            "Missing OPENROUTER_API_KEY. "
            "Paste it into scripts/run_openrouter_review_queue_pilot.py, "
            "or pass --api_key, or set the environment variable."
        )
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = api_key

    generate_cmd = [
        python_exe,
        "scripts/generate_vqa_labels.py",
        "--objects",
        "data/processed/metadata/objects_review_queue.jsonl",
        "--processed_dir",
        "data/processed",
        "--prompt",
        "prompts/vqa_traffic_vi.txt",
        "--out",
        "data/processed/review/raw_qa_openrouter_review_queue_pilot.jsonl",
        "--failed_out",
        "data/processed/review/failed_generation_openrouter_review_queue_pilot.jsonl",
        "--splits",
        "all",
        "--provider",
        "openrouter",
        "--model",
        args.model,
        "--limit",
        str(args.limit),
        "--temperature",
        "0.2",
        "--max_output_tokens",
        "4096",
        "--delay",
        "2",
        "--retries",
        "3",
        "--retry_delay",
        "5",
        "--rate_limit_sleep",
        "60",
        "--resume",
        "--stop_on_rate_limit",
    ]

    filter_cmd = [
        python_exe,
        "scripts/filter_vqa.py",
        "--raw",
        "data/processed/review/raw_qa_openrouter_review_queue_pilot.jsonl",
        "--out_jsonl",
        "data/processed/review/filtered_qa_openrouter_review_queue_pilot.jsonl",
        "--out_csv",
        "data/processed/review/review_openrouter_review_queue_pilot.csv",
        "--max_per_image",
        "10",
        "--min_per_image",
        "3",
        "--max_yes_no_ratio",
        "0.35",
    ]

    run_command(generate_cmd, env)
    run_command(filter_cmd, env)

    print("Done.")
    print("Review file: data/processed/review/review_openrouter_review_queue_pilot.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
