import os
import sys
import argparse
import math
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import ConfigB
from data_utils.dataset import load_vqa_data
from models.model_b import VQAModelB


class BlipVQADataset(Dataset):
    def __init__(self, samples, model_b, max_question_length=64,
                 max_answer_length=20):
        self.samples = samples
        self.model_b = model_b
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self.model_b.prepare_training_input(
            sample["image"],
            sample["question"],
            sample["answer"],
            max_question_length=self.max_question_length,
            max_answer_length=self.max_answer_length,
        )


class RawVQADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def select_stratified_samples(samples, limit, seed):
    target = min(int(limit), len(samples))
    if target <= 0:
        return []

    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in samples:
        key = sample.get("question_type", sample.get("type", "unknown"))
        groups[key].append(sample)
    for items in groups.values():
        rng.shuffle(items)

    selected = []
    keys = sorted(groups)
    while len(selected) < target and keys:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < target:
                selected.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    max_label_len = max(b["labels"].size(0) for b in batch)
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    decoder_attention_mask = []
    labels = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        label_pad_len = max_label_len - b["labels"].size(0)
        input_ids.append(
            torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        decoder_input_ids.append(
            torch.cat([
                b["decoder_input_ids"],
                torch.zeros(label_pad_len, dtype=torch.long),
            ])
        )
        decoder_attention_mask.append(
            torch.cat([
                b["decoder_attention_mask"],
                torch.zeros(label_pad_len, dtype=torch.long),
            ])
        )
        labels.append(
            torch.cat([
                b["labels"],
                torch.full((label_pad_len,), -100, dtype=torch.long),
            ])
        )

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "decoder_input_ids": torch.stack(decoder_input_ids),
        "decoder_attention_mask": torch.stack(decoder_attention_mask),
        "labels": torch.stack(labels),
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, device,
                    config, epoch, grad_accum_steps):
    model.model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(dtype=torch.bfloat16, enabled=config.training.bf16):
            outputs = model.model(**batch)
            loss = outputs.loss / grad_accum_steps

        loss.backward()

        should_step = (
            (step + 1) % grad_accum_steps == 0
            or (step + 1) == len(dataloader)
        )
        if should_step:
            nn.utils.clip_grad_norm_(
                model.model.parameters(), config.training.gradient_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()
        num_batches += 1
        pbar.set_postfix(
            loss=f"{outputs.loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device, config):
    model.model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(dtype=torch.bfloat16, enabled=config.training.bf16):
            outputs = model.model(**batch)

        total_loss += outputs.loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--stratified-train-samples", type=int, default=None)
    parser.add_argument("--stratified-val-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--backend", type=str, choices=["blip", "qwen25", "paligemma2"],
        default=None,
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs). 0 = disabled.")
    parser.add_argument("--wandb-project", type=str, default="sda-vqa")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = ConfigB()
    if args.data:
        config.vqa_data_path = args.data
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.grad_accum_steps is not None:
        config.training.gradient_accumulation_steps = args.grad_accum_steps
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.backend:
        config.model.backend = args.backend
    if args.model_name:
        config.model.model_name = args.model_name
    elif config.model.backend == "qwen25":
        config.model.model_name = config.model.qwen_model_name
    elif config.model.backend == "paligemma2":
        config.model.model_name = config.model.paligemma_model_name
    if args.load_in_4bit:
        config.model.load_in_4bit = True
    if args.min_pixels is not None:
        config.model.min_pixels = args.min_pixels
    if args.max_pixels is not None:
        config.model.max_pixels = args.max_pixels
    if args.max_length is not None:
        config.data.max_length = args.max_length
    if args.no_bf16:
        config.training.bf16 = False
    if config.model.backend == "paligemma2" and config.model.load_in_4bit and config.training.bf16:
        print("PaliGemma 2 with 4-bit detected; disabling bf16 autocast for stability.")
        config.training.bf16 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_labels = {
        "blip": "BLIP VQA",
        "qwen25": "Qwen2.5-VL",
        "paligemma2": "PaliGemma 2",
    }
    backend_label = backend_labels[config.model.backend]
    print(f"Training B2 ({backend_label} + LoRA) on {device}")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or f"b2_{config.model.backend}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": "b2",
                "backend": config.model.backend,
                "base_model": config.model.model_name,
                "lora_r": config.model.lora.r,
                "lora_alpha": config.model.lora.lora_alpha,
                "lora_dropout": config.model.lora.lora_dropout,
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "grad_accum_steps": config.training.gradient_accumulation_steps,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "warmup_ratio": config.training.warmup_ratio,
                "bf16": config.training.bf16,
            },
        )

    model_b = VQAModelB(
        model_name=config.model.model_name,
        lora_config=config.model.lora,
        device=device,
        backend=config.model.backend,
        load_in_4bit=config.model.load_in_4bit,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
    )
    device = model_b.device

    if config.model.backend in ("qwen25", "paligemma2") and config.data.num_workers != 0:
        print(
            f"{backend_label} backend uses processor-based batch collation; "
            "forcing num_workers=0"
        )
        config.data.num_workers = 0

    train_samples = load_vqa_data(config.vqa_data_path, "train")
    val_samples = load_vqa_data(config.vqa_data_path, "val")
    if args.stratified_train_samples is not None:
        train_samples = select_stratified_samples(
            train_samples,
            args.stratified_train_samples,
            args.sample_seed,
        )
    elif args.max_train_samples is not None:
        train_samples = train_samples[:args.max_train_samples]
    if args.stratified_val_samples is not None:
        val_samples = select_stratified_samples(
            val_samples,
            args.stratified_val_samples,
            args.sample_seed,
        )
    elif args.max_val_samples is not None:
        val_samples = val_samples[:args.max_val_samples]
    print(f"Loaded samples: train={len(train_samples)} val={len(val_samples)}")

    if config.model.backend in ("qwen25", "paligemma2"):
        train_dataset = RawVQADataset(train_samples)
        val_dataset = RawVQADataset(val_samples)

        def active_collate_fn(batch):
            return model_b.prepare_training_batch(
                batch,
                max_length=config.data.max_length,
            )
    else:
        train_dataset = BlipVQADataset(
            train_samples,
            model_b,
            max_question_length=config.data.max_question_length,
            max_answer_length=config.data.max_answer_length,
        )
        val_dataset = BlipVQADataset(
            val_samples,
            model_b,
            max_question_length=config.data.max_question_length,
            max_answer_length=config.data.max_answer_length,
        )
        active_collate_fn = collate_fn

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        collate_fn=active_collate_fn, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        collate_fn=active_collate_fn, pin_memory=(device.type == "cuda"),
    )

    trainable_params = [p for p in model_b.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_loader) / config.training.gradient_accumulation_steps
        ),
    )
    total_steps = steps_per_epoch * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6,
    )

    best_val_loss = float("inf")
    no_improve = 0
    patience = args.patience
    ckpt_names = {
        "blip": "model_b2",
        "qwen25": "model_b2_qwen25",
        "paligemma2": "model_b2_paligemma2",
    }
    ckpt_name = ckpt_names[config.model.backend]
    ckpt_dir = os.path.join(config.checkpoint_dir, ckpt_name)

    for epoch in range(1, config.training.epochs + 1):
        train_loss = train_one_epoch(
            model_b, train_loader, optimizer, scheduler, device,
            config, epoch, config.training.gradient_accumulation_steps,
        )
        val_loss = validate(model_b, val_loader, device, config)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model_b.save_lora(os.path.join(ckpt_dir, "best_lora"))
            if use_wandb:
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["best_epoch"] = epoch
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model_b.save_lora(os.path.join(ckpt_dir, "last_lora"))
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
