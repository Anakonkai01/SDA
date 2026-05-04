import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_b import VQAModelB
from train.config import ConfigB


class BlipPreferenceDataset(Dataset):
    def __init__(self, path, model_b, max_question_length=64,
                 max_answer_length=20):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.model_b = model_b
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        common = {
            "image_path": row["image"],
            "question": row["question"],
        }
        chosen = self.model_b.prepare_training_input(
            common["image_path"],
            common["question"],
            row["chosen"],
            max_question_length=self.max_question_length,
            max_answer_length=self.max_answer_length,
        )
        rejected = self.model_b.prepare_training_input(
            common["image_path"],
            common["question"],
            row["rejected"],
            max_question_length=self.max_question_length,
            max_answer_length=self.max_answer_length,
        )
        return {"chosen": chosen, "rejected": rejected}


def pad_1d(items, key, pad_value=0):
    max_len = max(item[key].size(0) for item in items)
    output = []
    for item in items:
        tensor = item[key]
        pad_len = max_len - tensor.size(0)
        if pad_len:
            tensor = torch.cat([
                tensor,
                torch.full((pad_len,), pad_value, dtype=tensor.dtype),
            ])
        output.append(tensor)
    return torch.stack(output)


def collate_side(items):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "input_ids": pad_1d(items, "input_ids", 0),
        "attention_mask": pad_1d(items, "attention_mask", 0),
        "decoder_input_ids": pad_1d(items, "decoder_input_ids", 0),
        "decoder_attention_mask": pad_1d(items, "decoder_attention_mask", 0),
        "labels": pad_1d(items, "labels", -100),
    }


def collate_fn(batch):
    return {
        "chosen": collate_side([item["chosen"] for item in batch]),
        "rejected": collate_side([item["rejected"] for item in batch]),
    }


def move_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def sequence_logps(model, batch, use_bf16=False):
    blip = model.get_base_model() if hasattr(model, "get_base_model") else model
    with autocast(dtype=torch.bfloat16, enabled=use_bf16):
        vision_outputs = blip.vision_model(
            pixel_values=batch["pixel_values"],
        )
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        question_outputs = blip.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        question_embeds = question_outputs[0]

        outputs = blip.text_decoder(
            input_ids=batch["decoder_input_ids"],
            attention_mask=batch["decoder_attention_mask"],
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits

    labels = batch["labels"]
    target = labels[:, 1:].contiguous()
    pred_logits = logits[:, :-1, :].contiguous()
    mask = target != -100
    safe_target = target.masked_fill(~mask, 0)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_logps = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    return losses.mean()


def main():
    parser = argparse.ArgumentParser(description="Train B2-DPO on preference pairs.")
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--sft-lora-path", default=None)
    parser.add_argument("--output-dir", default="checkpoints/model_b2_dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    config = ConfigB()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = VQAModelB(
        model_name=config.model.model_name,
        lora_config=config.model.lora,
        device=device,
    )
    if args.sft_lora_path:
        policy.load_lora(args.sft_lora_path)
        policy.model.train()

    reference = VQAModelB(
        model_name=config.model.model_name,
        lora_config=None,
        device=device,
    )
    if args.sft_lora_path:
        reference.load_lora(args.sft_lora_path)
    reference.model.eval()
    for param in reference.model.parameters():
        param.requires_grad = False

    dataset = BlipPreferenceDataset(
        args.preferences,
        policy,
        max_question_length=config.data.max_question_length,
        max_answer_length=config.data.max_answer_length,
    )
    if args.max_samples is not None:
        dataset.rows = dataset.rows[:args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    use_bf16 = not args.no_bf16

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        policy.model.train()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"DPO epoch {epoch}"):
            chosen = move_to_device(batch["chosen"], device)
            rejected = move_to_device(batch["rejected"], device)

            policy_chosen = sequence_logps(policy.model, chosen, use_bf16)
            policy_rejected = sequence_logps(policy.model, rejected, use_bf16)
            with torch.no_grad():
                ref_chosen = sequence_logps(reference.model, chosen, use_bf16)
                ref_rejected = sequence_logps(reference.model, rejected, use_bf16)

            loss = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta=args.beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"epoch={epoch} dpo_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            policy.save_lora(Path(args.output_dir) / "best_lora")

    policy.save_lora(Path(args.output_dir) / "last_lora")
    print(f"Training complete. Best DPO loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
