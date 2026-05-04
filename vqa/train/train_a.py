import os
import sys
import argparse
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import CLIPProcessor, AutoTokenizer
from tqdm import tqdm
import wandb

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


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    val_loss = float(ckpt.get("val_loss", float("inf")))
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(epoch={epoch}, val_loss={val_loss:.4f})"
    )
    return epoch, val_loss


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder", type=str, default="lstm", choices=["lstm", "transformer"],
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--stratified-train-samples", type=int, default=None)
    parser.add_argument("--stratified-val-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Checkpoint path to resume model weights from. Optimizer is rebuilt.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint_dir/model_<variant>/last.pt if present, else best.pt.",
    )
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs). 0 = disabled.")
    parser.add_argument("--wandb-project", type=str, default="sda-vqa")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = ConfigA()
    if args.data:
        config.vqa_data_path = args.data
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.phase1_epochs is not None:
        config.training.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        config.training.phase2_epochs = args.phase2_epochs
    config.training.epochs = (
        config.training.phase1_epochs + config.training.phase2_epochs
    )
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.no_mixed_precision:
        config.training.mixed_precision = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_type = args.decoder
    variant = "a1" if decoder_type == "lstm" else "a2"
    print(f"Training {variant.upper()} ({decoder_type} decoder) on {device}")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or variant
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": variant,
                "decoder": decoder_type,
                "phase1_epochs": config.training.phase1_epochs,
                "phase2_epochs": config.training.phase2_epochs,
                "batch_size": config.training.batch_size,
                "phase1_lr": config.training.phase1_lr,
                "phase2_lr": config.training.phase2_lr,
                "weight_decay": config.training.weight_decay,
                "mixed_precision": config.training.mixed_precision,
                "co_attention_layers": config.model.co_attention_layers,
                "lstm_layers": config.model.lstm_layers,
                "transformer_layers": config.model.transformer_layers,
                "image_encoder": config.model.image_encoder,
                "text_encoder": config.model.text_encoder,
            },
        )

    clip_processor = CLIPProcessor.from_pretrained(config.model.image_encoder)
    phobert_tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder)

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
    last_val_loss = float("inf")
    ckpt_dir = os.path.join(config.checkpoint_dir, f"model_{variant}")
    resume_path = args.resume_from
    if args.resume and resume_path is None:
        last_path = os.path.join(ckpt_dir, "last.pt")
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(last_path):
            resume_path = last_path
        elif os.path.exists(best_path):
            resume_path = best_path

    start_epoch = 1
    if resume_path:
        loaded_epoch, loaded_val_loss = load_checkpoint(model, resume_path, device)
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_val_loss
        last_val_loss = loaded_val_loss
        if start_epoch > config.training.epochs:
            print(
                f"Checkpoint epoch {loaded_epoch} already reaches configured "
                f"total epochs {config.training.epochs}."
            )
            return

    patience = args.patience
    no_improve = 0

    # Phase 1: freeze encoders
    print(f"\n=== Phase 1: Frozen encoders ({config.training.phase1_epochs} epochs) ===")
    stopped_early = False
    if config.training.phase1_epochs > 0 and start_epoch <= config.training.phase1_epochs:
        model.freeze_encoders()
        remaining_p1_epochs = config.training.phase1_epochs - start_epoch + 1
        num_steps_p1 = len(train_loader) * remaining_p1_epochs
        optimizer, scheduler = get_optimizer_and_scheduler(
            model, config, num_steps_p1, phase=1,
        )

        for epoch in range(start_epoch, config.training.phase1_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                criterion, device, config, epoch,
            )
            last_val_loss = validate(model, val_loader, criterion, device, config)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={last_val_loss:.4f}"
            )
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "phase": 1,
                    "train/loss": train_loss,
                    "val/loss": last_val_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                })

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                no_improve = 0
                save_checkpoint(
                    model, optimizer, epoch, last_val_loss,
                    os.path.join(ckpt_dir, "best.pt"),
                )
                if use_wandb:
                    wandb.summary["best_val_loss"] = best_val_loss
                    wandb.summary["best_epoch"] = epoch
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    stopped_early = True
                    break

    # Phase 2: unfreeze encoders with smaller lr
    print(f"\n=== Phase 2: Unfrozen encoders ({config.training.phase2_epochs} epochs) ===")
    no_improve = 0  # reset counter for phase 2
    if config.training.phase2_epochs > 0 and not stopped_early:
        model.unfreeze_encoders()
        phase2_start = max(start_epoch, config.training.phase1_epochs + 1)
        if phase2_start > config.training.epochs:
            print("No remaining phase 2 epochs to run.")
        num_steps_p2 = len(train_loader) * max(
            config.training.epochs - phase2_start + 1, 1
        )
        optimizer, scheduler = get_optimizer_and_scheduler(
            model, config, num_steps_p2, phase=2,
        )

        for epoch in range(phase2_start, config.training.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                criterion, device, config, epoch,
            )
            last_val_loss = validate(model, val_loader, criterion, device, config)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={last_val_loss:.4f}"
            )
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "phase": 2,
                    "train/loss": train_loss,
                    "val/loss": last_val_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                })

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                no_improve = 0
                save_checkpoint(
                    model, optimizer, epoch, last_val_loss,
                    os.path.join(ckpt_dir, "best.pt"),
                )
                if use_wandb:
                    wandb.summary["best_val_loss"] = best_val_loss
                    wandb.summary["best_epoch"] = epoch
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

    save_checkpoint(
        model, optimizer, config.training.epochs, last_val_loss,
        os.path.join(ckpt_dir, "last.pt"),
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
