#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="stable_diffusion/logs"
LOG_FILE="$LOG_DIR/overnight_lora_${TIMESTAMP}.log"
DATA_FILE="stable_diffusion/dataset/metadata.jsonl"
OUTPUT_DIR="stable_diffusion/lora_output"
SAMPLE_DIR="stable_diffusion/outputs/overnight_${TIMESTAMP}"

DATA_LIMIT="${DATA_LIMIT:-1500}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-12000}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-7h}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
POWER_ACTION="${POWER_ACTION:-suspend}"

mkdir -p "$LOG_DIR" "stable_diffusion/dataset" "$SAMPLE_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Stable Diffusion LoRA overnight run ==="
echo "Start: $(date)"
echo "Root: $ROOT"
echo "GPU status:"
nvidia-smi || true

echo "=== Step 1: Build LoRA metadata dataset ==="
python stable_diffusion/build_lora_dataset.py \
  --limit "$DATA_LIMIT" \
  --output "$DATA_FILE"

echo "=== Step 2: Train LoRA ==="
set +e
timeout "$TRAIN_TIMEOUT" python stable_diffusion/train_lora.py \
  --data "$DATA_FILE" \
  --image-root data/processed \
  --output-dir "$OUTPUT_DIR" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --resolution 512 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --mixed-precision "$MIXED_PRECISION" \
  --learning-rate "$LEARNING_RATE" \
  --save-every "$SAVE_EVERY"
TRAIN_EXIT=$?
set -e

if [[ "$TRAIN_EXIT" -eq 124 ]]; then
  echo "Training reached timeout $TRAIN_TIMEOUT; using latest checkpoint if final adapter was not written."
elif [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "Training failed with exit code $TRAIN_EXIT. Continuing to final logging; sample generation may be skipped."
fi

if [[ ! -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]]; then
  LATEST_CHECKPOINT="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
  if [[ -n "$LATEST_CHECKPOINT" && -f "$LATEST_CHECKPOINT/pytorch_lora_weights.safetensors" ]]; then
    echo "Copying latest checkpoint $LATEST_CHECKPOINT to $OUTPUT_DIR"
    cp "$LATEST_CHECKPOINT/pytorch_lora_weights.safetensors" "$OUTPUT_DIR/pytorch_lora_weights.safetensors"
  fi
fi

echo "=== Step 3: Generate base vs LoRA sample images ==="
python - <<'PY'
import importlib.util
from pathlib import Path

root = Path.cwd()
app_path = root / "stable_diffusion" / "app.py"
sample_dir = Path("stable_diffusion/outputs")
spec = importlib.util.spec_from_file_location("sd_app", app_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

prompts = [
    "a realistic Vietnamese street intersection with traffic signs, daytime, high detail",
    "a road safety education poster about Vietnamese traffic signs, clean composition",
    "Vietnamese traffic road scene, real street photo, multiple traffic signs, high detail",
]

for i, prompt in enumerate(prompts, 1):
    for model_choice in ["Base pretrained", "Fine-tuned LoRA"]:
        image, metadata = mod.generate(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, unreadable text",
            steps=25,
            guidance_scale=7.5,
            seed=1000 + i,
            size=512,
            model_choice=model_choice,
        )
        label = "lora" if model_choice.startswith("Fine") else "base"
        out = sample_dir / f"overnight_{i}_{label}.png"
        image.save(out)
        (sample_dir / f"overnight_{i}_{label}.txt").write_text(metadata, encoding="utf-8")
        print(f"Saved {out}")
PY

echo "=== Step 4: Final status ==="
echo "End: $(date)"
echo "Log: $LOG_FILE"
echo "LoRA output: $OUTPUT_DIR"
echo "Samples: stable_diffusion/outputs/overnight_*.png"
nvidia-smi || true

case "$POWER_ACTION" in
  suspend)
    echo "Suspending machine now because POWER_ACTION=suspend."
    sync
    systemctl suspend
    ;;
  shutdown)
    echo "Shutting down machine now because POWER_ACTION=shutdown."
    sync
    shutdown -h now
    ;;
  none)
    echo "Skipping power action because POWER_ACTION=none."
    ;;
  *)
    echo "Unknown POWER_ACTION=$POWER_ACTION; skipping power action."
    ;;
esac
