import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import CLIPProcessor, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import ConfigA
from data_utils.dataset import VQADataset, load_vqa_data
from data_utils.collator import VQACollator
from models.model_a import VQAModelA


def get_optimizer_and_scheduler(model, config, num_training_steps, phase=1):
    if phase == 1:
        lr = config.training.phase1_lr
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        lr = config.training.phase2_lr
        encoder_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "image_encoder" in name or "text_encoder" in name:
                encoder_params.append(p)
            else:
                other_params.append(p)
        params = [
            {"params": encoder_params, "lr": lr},
            {"params": other_params, "lr": lr * 10},
        ]

    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=config.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps, eta_min=1e-6,
    )

    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                    criterion, device, config, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with autocast(dtype=torch.float16, enabled=config.training.mixed_precision):
            logits = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                batch["labels"].reshape(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(dtype=torch.float16, enabled=config.training.mixed_precision):
            logits = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                batch["labels"].reshape(-1),
            )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder", type=str, default="lstm", choices=["lstm", "transformer"],
    )
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    config = ConfigA()
    if args.data:
        config.vqa_data_path = args.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_type = args.decoder
    variant = "a1" if decoder_type == "lstm" else "a2"
    print(f"Training {variant.upper()} ({decoder_type} decoder) on {device}")

    clip_processor = CLIPProcessor.from_pretrained(config.model.image_encoder)
    phobert_tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder)

    train_samples = load_vqa_data(config.vqa_data_path, "train")
    val_samples = load_vqa_data(config.vqa_data_path, "val")

    train_dataset = VQADataset(
        train_samples, clip_processor, phobert_tokenizer,
        max_q_len=config.data.max_question_length,
        max_a_len=config.data.max_answer_length,
    )
    val_dataset = VQADataset(
        val_samples, clip_processor, phobert_tokenizer,
        max_q_len=config.data.max_question_length,
        max_a_len=config.data.max_answer_length,
    )

    collator = VQACollator()
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        collate_fn=collator, pin_memory=True,
    )

    model = VQAModelA(
        decoder_type=decoder_type,
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        clip_dim=config.model.clip_dim,
        co_attn_layers=config.model.co_attention_layers,
        co_attn_heads=config.model.co_attention_heads,
        dropout=config.model.dropout,
        lstm_layers=config.model.lstm_layers,
        transformer_layers=config.model.transformer_layers,
        transformer_ffn=config.model.transformer_ffn,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler(enabled=config.training.mixed_precision)

    best_val_loss = float("inf")
    ckpt_dir = os.path.join(config.checkpoint_dir, f"model_{variant}")

    # Phase 1: freeze encoders
    print(f"\n=== Phase 1: Frozen encoders ({config.training.phase1_epochs} epochs) ===")
    model.freeze_encoders()
    num_steps_p1 = len(train_loader) * config.training.phase1_epochs
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, config, num_steps_p1, phase=1,
    )

    for epoch in range(1, config.training.phase1_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, config, epoch,
        )
        val_loss = validate(model, val_loader, criterion, device, config)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(ckpt_dir, "best.pt"),
            )

    # Phase 2: unfreeze encoders with smaller lr
    print(f"\n=== Phase 2: Unfrozen encoders ({config.training.phase2_epochs} epochs) ===")
    model.unfreeze_encoders()
    num_steps_p2 = len(train_loader) * config.training.phase2_epochs
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, config, num_steps_p2, phase=2,
    )

    for epoch in range(
        config.training.phase1_epochs + 1,
        config.training.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, config, epoch,
        )
        val_loss = validate(model, val_loader, criterion, device, config)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(ckpt_dir, "best.pt"),
            )

    save_checkpoint(
        model, optimizer, config.training.epochs, val_loss,
        os.path.join(ckpt_dir, "last.pt"),
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
