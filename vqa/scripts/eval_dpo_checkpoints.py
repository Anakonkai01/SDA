#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def checkpoint_step(path):
    match = re.search(r"checkpoint_(\d+)pairs", path.name)
    return int(match.group(1)) if match else 10**12


def read_accuracy(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("B2_Qwen25_LoRA") or next(iter(data.values()))
    return float(metrics.get("vqa_accuracy", 0.0)), metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO checkpoints on the same stratified subset.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", default="reports/dpo_checkpoint_eval")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-pixels", type=int, default=501760)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--include-best-last", action="store_true")
    args = parser.parse_args()

    root = Path(args.checkpoint_dir)
    checkpoints = sorted(root.glob("checkpoint_*pairs"), key=checkpoint_step)
    if args.include_best_last:
        checkpoints.extend(p for p in [root / "best_lora", root / "last_lora"] if p.exists())
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for ckpt in checkpoints:
        tag = ckpt.name
        result_path = out_dir / f"{tag}_results.json"
        pred_path = out_dir / f"{tag}_predictions.jsonl"
        cmd = [
            sys.executable, "evaluate/evaluate.py",
            "--model", "b2",
            "--backend", "qwen25",
            "--load-in-4bit",
            "--max-pixels", str(args.max_pixels),
            "--lora-path", str(ckpt),
            "--stratified-limit", str(args.limit),
            "--stratified-seed", str(args.seed),
            "--data-split", args.data_split,
            "--eval-batch-size", str(args.eval_batch_size),
            "--output", str(result_path),
            "--predictions-output", str(pred_path),
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
        acc, metrics = read_accuracy(result_path)
        summary.append({"checkpoint": str(ckpt), "accuracy": acc, "result": str(result_path)})
        print(f"DONE {tag}: accuracy={acc:.4f}", flush=True)

    summary.sort(key=lambda row: row["accuracy"], reverse=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nBest checkpoints")
    for row in summary[:10]:
        print(f"{row['accuracy']:.4f}  {row['checkpoint']}")


if __name__ == "__main__":
    main()