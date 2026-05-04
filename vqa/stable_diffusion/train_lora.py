import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "runwayml/stable-diffusion-v1-5"


class CaptionImageDataset(Dataset):
    def __init__(self, metadata_path, image_root, tokenizer, resolution):
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.rows = []
        with Path(metadata_path).open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                image_path = self.image_root / row["image"]
                if image_path.exists():
                    self.rows.append({"image": image_path, "caption": row["caption"]})
        if not self.rows:
            raise ValueError(f"No valid images found from {metadata_path} under {image_root}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        pixel_values = self.transform(image)
        tokenized = self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with a small LoRA adapter.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "stable_diffusion" / "lora_output")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unwrap_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this LoRA training script.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    weight_dtype = unwrap_dtype(args.mixed_precision)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device, dtype=weight_dtype)
    vae = pipe.vae.to(device, dtype=weight_dtype)
    unet = pipe.unet.to(device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    try:
        from peft import LoraConfig, get_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA training. Install requirements.txt first.") from exc

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    dataset = CaptionImageDataset(args.data, args.image_root, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def save_lora(path):
        path.mkdir(parents=True, exist_ok=True)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=path,
            unet_lora_layers=get_peft_model_state_dict(unet),
        )

    progress = tqdm(total=args.max_train_steps, desc="Training LoRA")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite; lower --learning-rate or use --mixed-precision no.")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

            if global_step % args.save_every == 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                save_lora(checkpoint_dir)

            if global_step >= args.max_train_steps:
                break

    progress.close()
    save_lora(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
