"""
Upload VQA dataset and model checkpoints to HuggingFace Hub.
Run from vqa/ directory after: huggingface-cli login

Usage:
    python scripts/upload_to_hf.py --what dataset
    python scripts/upload_to_hf.py --what models
    python scripts/upload_to_hf.py --what all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def upload_dataset(api, username: str):
    from huggingface_hub import DatasetCard

    repo_id = f"{username}/vietnamese-traffic-sign-vqa"
    print(f"\n=== Uploading dataset → {repo_id} ===")

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    # Upload annotation JSONL files
    ann_dir = ROOT / "data" / "processed" / "annotations"
    for f in sorted(ann_dir.glob("*.jsonl")):
        print(f"  Uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"annotations/{f.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload objects metadata
    obj_file = ROOT / "data" / "processed" / "metadata" / "objects.jsonl"
    if obj_file.exists():
        print(f"  Uploading objects.jsonl ({obj_file.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(obj_file),
            path_in_repo="metadata/objects.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Dataset card
    card_content = """---
language:
- vi
license: cc-by-sa-4.0
task_categories:
- visual-question-answering
tags:
- vietnamese
- traffic-signs
- vqa
pretty_name: Vietnamese Traffic Sign VQA
---

# Vietnamese Traffic Sign VQA

Visual Question Answering dataset for Vietnamese traffic signs.
Built from [Kaggle VNTS](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs) (CC BY-SA 4.0).

## Statistics

| Split | Images | QA Pairs | QA/Image |
|---|---:|---:|---:|
| Train | 2,193 | 104,146 | 47.5 |
| Val | 272 | 12,944 | 47.6 |
| Test | 271 | 12,966 | 47.8 |
| **Total** | **2,736** | **130,056** | **47.5** |

## Question Types
12 types: `yes_no`, `count`, `sign_type`, `color`, `shape`, `location`, `attribute`, `negative`, `spatial_rel`, `count_total`, `multi_object`, `context`

## Format
```json
{
  "question_id": "vts_000001_q01",
  "image_id": "vts_000001",
  "image_path": "images/train/vts_000001.jpg",
  "question": "Trong ảnh có biển giới hạn tốc độ không?",
  "answer": "Có",
  "question_type": "yes_no",
  "split": "train"
}
```

## Models
See [`Anakonkai/vietnamese-traffic-sign-vqa-checkpoints`](https://huggingface.co/Anakonkai/vietnamese-traffic-sign-vqa-checkpoints)
"""
    card_path = ROOT / "data" / "processed" / "annotations" / "README_HF.md"
    card_path.write_text(card_content, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    card_path.unlink()

    print(f"✅ Dataset uploaded → https://huggingface.co/datasets/{repo_id}")


def upload_models(api, username: str):
    repo_id = f"{username}/vietnamese-traffic-sign-vqa-checkpoints"
    print(f"\n=== Uploading model checkpoints → {repo_id} ===")

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    checkpoints = [
        {
            "local": ROOT / "checkpoints_v8_a_50k" / "model_a1" / "best.pt",
            "remote": "a1_lstm/best.pt",
            "label": "A1 LSTM decoder checkpoint",
        },
        {
            "local": ROOT / "checkpoints_v8_a_50k" / "model_a2" / "best.pt",
            "remote": "a2_transformer/best.pt",
            "label": "A2 Transformer decoder checkpoint",
        },
        {
            "local": ROOT / "checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5" / "model_b2_qwen25" / "best_lora",
            "remote": "b2_sft_lora/",
            "label": "B2-SFT QLoRA adapter (best)",
            "is_dir": True,
        },
        {
            "local": ROOT / "checkpoints_b2_dpo_balanced_v1" / "checkpoint_500pairs",
            "remote": "b2_dpo_lora/",
            "label": "B2-DPO balanced checkpoint (500 pairs, best VQA acc)",
            "is_dir": True,
        },
        {
            "local": ROOT / "stable_diffusion" / "lora_output" / "pytorch_lora_weights.safetensors",
            "remote": "stable_diffusion_lora/pytorch_lora_weights.safetensors",
            "label": "Stable Diffusion LoRA adapter (12k steps, 1500 traffic sign images)",
        },
    ]

    for ckpt in checkpoints:
        local = Path(ckpt["local"])
        if not local.exists():
            print(f"  ⚠️  SKIP (not found): {local}")
            continue

        if ckpt.get("is_dir"):
            print(f"  Uploading folder {ckpt['label']} ...")
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=ckpt["remote"],
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            size_mb = local.stat().st_size / 1e6
            print(f"  Uploading {ckpt['label']} ({size_mb:.0f} MB)...")
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=ckpt["remote"],
                repo_id=repo_id,
                repo_type="model",
            )

    # Model card
    card_content = f"""---
language:
- vi
license: apache-2.0
tags:
- visual-question-answering
- vietnamese
- traffic-signs
- lora
- qwen2-vl
- stable-diffusion
---

# Vietnamese Traffic Sign VQA — Checkpoints

Model checkpoints for the Deep Learning course final project.

## Contents

| Path | Description | Size |
|---|---|---|
| `a1_lstm/best.pt` | A1: CLIP + PhoBERT + Co-Attention + **LSTM** decoder | ~2.4 GB |
| `a2_transformer/best.pt` | A2: CLIP + PhoBERT + Co-Attention + **Transformer** decoder | ~2.5 GB |
| `b2_sft_lora/` | B2-SFT: Qwen2.5-VL-3B **QLoRA** adapter (best checkpoint) | ~153 MB |
| `b2_dpo_lora/` | B2-DPO: Qwen2.5-VL-3B DPO-aligned adapter (500 pairs, balanced) | ~153 MB |
| `stable_diffusion_lora/` | SD LoRA: Stable Diffusion v1.5 LoRA on traffic sign images (12k steps) | ~6 MB |

## Results (Full Test Set)

| Model | VQA Acc | BLEU-4 | ROUGE-L | BERTScore | Latency |
|---|---:|---:|---:|---:|---:|
| A1 LSTM | **0.9484** | **0.9602** | **0.9584** | 0.9631 | ~11 ms |
| A2 Transformer | 0.9377 | 0.9476 | 0.9486 | **0.9713** | ~13 ms |
| B1 Zero-shot | 0.1962 | 0.0350 | 0.2899 | 0.4753 | ~167 ms |
| B2-SFT QLoRA | 0.9379 | 0.9494 | 0.9508 | 0.9111 | ~484 ms |

Human evaluation (100 natural questions): B2-DPO **73%** > B2-SFT 70% > A1 61%

## Dataset
See [`{username}/vietnamese-traffic-sign-vqa`](https://huggingface.co/datasets/{username}/vietnamese-traffic-sign-vqa)

## Code
GitHub: https://github.com/{username}/SDA

## Usage

### A1 / A2
```python
import torch
from models.model_a import VQAModelA

model = VQAModelA(decoder_type="lstm")  # or "transformer"
ckpt = torch.load("a1_lstm/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
```

### B2-SFT
```bash
python evaluate/evaluate.py \\
  --model b2 --backend qwen25 --load-in-4bit \\
  --checkpoint b2_sft_lora/ \\
  --data data/processed/annotations
```

### Stable Diffusion LoRA
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("stable_diffusion_lora/")
image = pipe("a Vietnamese street intersection with traffic signs").images[0]
```
"""
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(card_content)
        tmp = f.name
    api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    os.unlink(tmp)

    print(f"✅ Models uploaded → https://huggingface.co/models/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["dataset", "models", "all"], default="all")
    parser.add_argument("--username", default="Anakonkai")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Verify login
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("ERROR: Not logged in. Run: huggingface-cli login")
        sys.exit(1)

    if args.what in ("dataset", "all"):
        upload_dataset(api, args.username)

    if args.what in ("models", "all"):
        upload_models(api, args.username)

    print("\n🎉 Done!")
    print(f"  Dataset : https://huggingface.co/datasets/{args.username}/vietnamese-traffic-sign-vqa")
    print(f"  Models  : https://huggingface.co/{args.username}/vietnamese-traffic-sign-vqa-checkpoints")


if __name__ == "__main__":
    main()
