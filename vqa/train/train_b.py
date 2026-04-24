import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import ConfigB
from data_utils.dataset import load_vqa_data
from models.model_b import VQAModelB


class QwenVQADataset(Dataset):
    def __init__(self, samples, model_b, max_length=256):
        self.samples = samples
        self.model_b = model_b
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self.model_b.prepare_training_input(
            sample["image"], sample["question"], sample["answer"],
        )


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(
            torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
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
            outputs = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
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
            outputs = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        total_loss += outputs.loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    config = ConfigB()
    if args.data:
        config.vqa_data_path = args.data

    device = torch.device("cuda")
    print(f"Training B2 (Qwen-VL + LoRA) on {device}")

    model_b = VQAModelB(
        model_name=config.model.model_name,
        lora_config=config.model.lora,
    )

    train_samples = load_vqa_data(config.vqa_data_path, "train")
    val_samples = load_vqa_data(config.vqa_data_path, "val")

    train_dataset = QwenVQADataset(train_samples, model_b, config.data.max_length)
    val_dataset = QwenVQADataset(val_samples, model_b, config.data.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    trainable_params = [p for p in model_b.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_steps = (
        len(train_loader) // config.training.gradient_accumulation_steps
        * config.training.epochs
    )
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    ckpt_dir = os.path.join(config.checkpoint_dir, "model_b2")

    for epoch in range(1, config.training.epochs + 1):
        train_loss = train_one_epoch(
            model_b, train_loader, optimizer, scheduler, device,
            config, epoch, config.training.gradient_accumulation_steps,
        )
        val_loss = validate(model_b, val_loader, device, config)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_b.save_lora(os.path.join(ckpt_dir, "best_lora"))

    model_b.save_lora(os.path.join(ckpt_dir, "last_lora"))
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
