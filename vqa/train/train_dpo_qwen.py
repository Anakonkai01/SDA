"""
DPO training for B2 (Qwen2.5-VL-3B-Instruct + LoRA).

Architecture:
  1. Compute reference logprobs (frozen SFT model, eval mode) → precompute once
  2. Free reference, load policy model (trainable LoRA)
  3. Train with standard DPO loss

Key optimization: reference logprobs precomputed to save VRAM (only 1 model on GPU during training).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import PeftModel
from models.model_b import VQAModelB
from train.config import ConfigB


class QwenVLPreferenceDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if max_samples is not None and max_samples < len(self.rows):
            groups = defaultdict(list)
            for r in self.rows:
                groups[r.get("question_type", "other")].append(r)
            sampled = []
            keys = sorted(groups)
            while len(sampled) < max_samples and keys:
                for k in keys:
                    if groups[k] and len(sampled) < max_samples:
                        sampled.append(groups[k].pop())
            self.rows = sampled[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = row["image"]
        if not img.startswith("/"):
            img = str(Path(__file__).parent.parent / img)
        return {
            "image": img,
            "image_id": row.get("image_id", ""),
            "question": row["question"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "question_type": row.get("question_type", ""),
            "ref_chosen_logp": row.get("ref_chosen_logp"),
            "ref_rejected_logp": row.get("ref_rejected_logp"),
        }

    def attach_reference_logps(self, chosen_logps, rejected_logps):
        if len(chosen_logps) != len(self.rows) or len(rejected_logps) != len(self.rows):
            raise ValueError("Reference logprob length mismatch.")
        for row, chosen, rejected in zip(self.rows, chosen_logps, rejected_logps):
            row["ref_chosen_logp"] = float(chosen)
            row["ref_rejected_logp"] = float(rejected)


def sequence_logps_qwen(model, batch, use_bf16=False):
    labels = batch["labels"]
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)

    forward_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch.get("pixel_values"),
        "image_grid_thw": batch.get("image_grid_thw"),
        "return_dict": True,
    }

    with torch.amp.autocast("cuda", enabled=False):
        outputs = model(**forward_kwargs)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = safe_labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * shift_mask
    token_counts = shift_mask.sum(dim=-1).clamp_min(1)
    # Length-normalized answer logprobs keep DPO margins comparable across short
    # and descriptive answers, and avoid hiding failures behind hard clamps.
    return token_logps.sum(dim=-1) / token_counts


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    logits = torch.clamp(logits, -10.0, 10.0)
    return -F.logsigmoid(logits).mean()


def collate_preferences(batch, model_b, max_length=2048):
    chosen_samples = [{"image": item["image"], "question": item["question"],
                       "answer": item["chosen"]} for item in batch]
    rejected_samples = [{"image": item["image"], "question": item["question"],
                         "answer": item["rejected"]} for item in batch]

    chosen_batch = model_b.prepare_training_batch(chosen_samples, max_length=max_length)
    rejected_batch = model_b.prepare_training_batch(rejected_samples, max_length=max_length)

    return {
        "chosen": {k: v.to(model_b.device) for k, v in chosen_batch.items()},
        "rejected": {k: v.to(model_b.device) for k, v in rejected_batch.items()},
        "ref_chosen": torch.tensor(
            [item.get("ref_chosen_logp", float("nan")) for item in batch],
            dtype=torch.float32,
            device=model_b.device,
        ),
        "ref_rejected": torch.tensor(
            [item.get("ref_rejected_logp", float("nan")) for item in batch],
            dtype=torch.float32,
            device=model_b.device,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="DPO for B2 (Qwen2.5-VL + LoRA).")
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--sft-lora-path", default=None)
    parser.add_argument("--output-dir", default="checkpoints/model_b2_dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=501760)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=250,
                        help="Save checkpoint every N pairs (0=disable)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle preference pairs after reference logprobs are attached.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = not args.no_bf16

    # ── Step 1: Dataset ──
    dataset = QwenVLPreferenceDataset(args.preferences, max_samples=args.max_samples)
    print(f"Preference pairs: {len(dataset)}")

    # ── Step 2: Reference model load → precompute → free ──
    print("Loading reference model (frozen)...")
    reference = VQAModelB(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        lora_config=None, device=device,
        max_pixels=args.max_pixels,
        backend="qwen25", load_in_4bit=True,
    )
    reference.model = PeftModel.from_pretrained(reference.model, args.sft_lora_path)
    reference.model.eval()
    for param in reference.model.parameters():
        param.requires_grad = False

    print(f"Precomputing reference logprobs...")
    ref_chosen_list, ref_rejected_list = [], []
    for i in tqdm(range(len(dataset)), desc="Ref logprobs"):
        batch = collate_preferences([dataset[i]], reference, max_length=args.max_length)
        with torch.no_grad():
            rc = sequence_logps_qwen(reference.model, batch["chosen"], use_bf16).cpu()
            rr = sequence_logps_qwen(reference.model, batch["rejected"], use_bf16).cpu()
        if not torch.isfinite(rc).all() or not torch.isfinite(rr).all():
            raise RuntimeError(f"Non-finite reference logprobs at pair {i}")
        ref_chosen_list.append(rc.item())
        ref_rejected_list.append(rr.item())
        del batch, rc, rr
    ref_chosen = torch.tensor(ref_chosen_list)
    ref_rejected = torch.tensor(ref_rejected_list)

    dataset.attach_reference_logps(ref_chosen.tolist(), ref_rejected.tolist())
    if args.shuffle:
        random.Random(args.seed).shuffle(dataset.rows)

    del reference, ref_chosen_list, ref_rejected_list
    torch.cuda.empty_cache()
    print("Reference freed.")

    # ── Step 3: Policy model ──
    print("Loading policy model (trainable)...")
    policy = VQAModelB(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        lora_config=None, device=device,
        max_pixels=args.max_pixels,
        backend="qwen25", load_in_4bit=True,
    )
    policy.model = PeftModel.from_pretrained(policy.model, args.sft_lora_path)
    from peft import prepare_model_for_kbit_training
    policy.model = prepare_model_for_kbit_training(policy.model)
    for name, param in policy.model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    for name, param in policy.model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
    if hasattr(policy.model, 'gradient_checkpointing_enable'):
        policy.model.gradient_checkpointing_enable()
    base = policy.model.get_base_model()
    if hasattr(base, 'gradient_checkpointing_enable'):
        base.gradient_checkpointing_enable()
    policy.model.config.use_cache = False
    policy.model.train()

    # ── Step 4: Dataloader ──
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_preferences(batch, policy, max_length=args.max_length),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    # ── Step 5: Train ──
    best_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        policy.model.train()
        total_loss, steps = 0.0, 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"DPO epoch {epoch}")):
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            policy_chosen = sequence_logps_qwen(policy.model, chosen, use_bf16)
            policy_rejected = sequence_logps_qwen(policy.model, rejected, use_bf16)

            ref_c = batch["ref_chosen"].to(policy_chosen.device)
            ref_r = batch["ref_rejected"].to(policy_chosen.device)
            if not torch.isfinite(ref_c).all() or not torch.isfinite(ref_r).all():
                raise RuntimeError("Missing/non-finite attached reference logprobs in batch.")

            loss = dpo_loss(policy_chosen, policy_rejected, ref_c, ref_r, beta=args.beta)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite DPO loss at epoch={epoch} step={batch_idx}; "
                    "stopping without saving a corrupted checkpoint."
                )

            optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip and args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.model.parameters() if p.requires_grad],
                    args.gradient_clip,
                )
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if steps == 1 or steps % 25 == 0:
                with torch.no_grad():
                    policy_margin = (policy_chosen - policy_rejected).mean().item()
                    ref_margin = (ref_c - ref_r).mean().item()
                    preferred = (policy_chosen > policy_rejected).float().mean().item()
                print(
                    f"  step={steps} loss={loss.item():.4f} "
                    f"policy_margin={policy_margin:.4f} ref_margin={ref_margin:.4f} "
                    f"chosen_pref={preferred:.2f}"
                )

            # Periodic checkpoint every N pairs
            if args.checkpoint_every > 0 and steps > 0 and steps % args.checkpoint_every == 0:
                pair_dir = output_dir / f"checkpoint_{steps}pairs"
                pair_dir.mkdir(parents=True, exist_ok=True)
                policy.model.save_pretrained(str(pair_dir))
                policy.processor.save_pretrained(str(pair_dir))
                print(f"  Saved checkpoint @ {steps} pairs → {pair_dir}")

        avg_loss = total_loss / max(steps, 1)
        print(f"epoch={epoch} dpo_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = output_dir / "best_lora"
            best_dir.mkdir(parents=True, exist_ok=True)
            policy.model.save_pretrained(str(best_dir))
            policy.processor.save_pretrained(str(best_dir))
            # Quick validation: generate 1 answer to catch corruption
            try:
                test_row = dataset.rows[0]
                test_batch = collate_preferences([dataset[0]], policy, max_length=args.max_length)
                with torch.no_grad():
                    test_out = policy.model.generate(
                        **{k: test_batch["chosen"][k] for k in ["input_ids","attention_mask","pixel_values","image_grid_thw"]},
                        max_new_tokens=5, do_sample=False)
                test_pred = policy.processor.decode(
                    test_out[0, test_batch["chosen"]["input_ids"].shape[1]:],
                    skip_special_tokens=True).strip()
                if len(test_pred) > 0 and not any(c * 5 in test_pred for c in set(test_pred)):
                    print(f"  Validation OK: '{test_pred[:30]}'")
                else:
                    print(f"  ⚠️ Validation degraded: '{test_pred[:30]}' — may be corrupted")
            except Exception as e:
                print(f"  ⚠️ Validation failed: {e}")
            print(f"  Saved → {best_dir}")

    last_dir = output_dir / "last_lora"
    last_dir.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(str(last_dir))
    print(f"Done. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
