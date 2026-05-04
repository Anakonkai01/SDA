import random
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_ROOT = ROOT / "stable_diffusion" / "lora_output_caption_en_12000"
BEST_LORA_DIR = LORA_ROOT / "checkpoint-3500"
OUTPUT_DIR = ROOT / "stable_diffusion" / "outputs"

pipe_cache = {"pipe": None, "adapter": None}

REALISTIC_PROMPTS = [
    "A realistic documentary photo of a Vietnamese urban street intersection with several visible traffic signs, motorbikes, cars, natural daylight, 35mm photography, high detail",
    "A realistic street photo in Vietnam with one red and white circular no-left-turn traffic sign near the roadside, motorbikes in the background, natural daylight, documentary photography style",
    "A realistic Vietnamese city road scene with yellow triangular warning traffic signs, trees, asphalt road, daylight, natural colors, high detail, shot on a phone camera",
    "A realistic close-up photo of a blue and white mandatory traffic sign on a Vietnamese street, shallow depth of field, natural daylight, documentary photo",
    "A realistic rainy Vietnamese street scene with traffic signs, wet asphalt, motorbikes, overcast daylight, natural photo, high detail",
    "A realistic Vietnamese street with a red and white speed limit traffic sign on the roadside, cars and motorbikes passing, natural daylight, urban environment",
]

DEFAULT_COMPARISON = [
    "Base pretrained",
    "checkpoint-500",
    "checkpoint-1000",
    "checkpoint-2000",
    "checkpoint-3000",
    "checkpoint-3500",
    "checkpoint-5000",
    "checkpoint-8000",
    "checkpoint-12000",
]


def device_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return "cuda", torch.float16, f"CUDA: {name} ({vram_gb:.1f} GB)"
    return "cpu", torch.float32, "CPU"


def adapter_exists(path):
    return path.exists() and (
        any(path.glob("*.safetensors")) or any(path.glob("pytorch_lora_weights.*"))
    )


def available_adapters():
    adapters = {"Base pretrained": None}
    if LORA_ROOT.exists():
        checkpoints = sorted(
            [p for p in LORA_ROOT.glob("checkpoint-*") if p.is_dir() and adapter_exists(p)],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 10**12,
        )
        for p in checkpoints:
            label = p.name
            if p == BEST_LORA_DIR:
                label = f"{p.name} ★ best visual"
            adapters[label] = p
        if adapter_exists(LORA_ROOT):
            adapters["final (12000)"] = LORA_ROOT
    return adapters


def checkpoint_sort_key(label):
    if label == "Base pretrained":
        return -1
    if "checkpoint-" in label:
        part = label.replace("★ best visual", "").strip().split("-")[-1]
        return int(part) if part.isdigit() else 10**8
    if "final" in label:
        return 10**9
    return 10**8


def model_choices():
    return list(available_adapters().keys())


def comparison_defaults():
    choices = model_choices()
    result = []
    for target in DEFAULT_COMPARISON:
        # match exact or with suffix like "★ best visual"
        match = next((c for c in choices if c == target or c.startswith(target + " ")), None)
        if match:
            result.append(match)
    return result


def load_pipeline(model_choice):
    adapters = available_adapters()
    lora_path = adapters.get(model_choice)

    if pipe_cache["pipe"] is None:
        device, dtype, _ = device_info()
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        pipe_cache["pipe"] = pipe

    pipe = pipe_cache["pipe"]
    adapter_key = str(lora_path) if lora_path is not None else None
    if pipe_cache["adapter"] != adapter_key:
        pipe.unload_lora_weights()
        if lora_path is not None:
            pipe.load_lora_weights(lora_path)
        pipe_cache["adapter"] = adapter_key
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pipe


def normalize_seed(seed):
    if seed is None or int(seed) < 0:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def generate_one(prompt, steps, guidance_scale, seed, size, model_choice):
    if not prompt or not prompt.strip():
        return None, "Vui lòng nhập prompt."

    selected_seed = normalize_seed(seed)
    pipe = load_pipeline(model_choice)
    device, _, device_text = device_info()
    generator = torch.Generator(device=device).manual_seed(selected_seed)

    image = pipe(
        prompt=prompt.strip(),
        negative_prompt=None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=int(size),
        height=int(size),
        generator=generator,
    ).images[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_choice.replace(" ", "_").replace("/", "-").replace("★", "best")
    output_path = OUTPUT_DIR / f"sd_{safe_model}_seed{selected_seed}_steps{steps}_cfg{guidance_scale:.1f}.png"
    image.save(output_path)

    info = (
        f"Model:   {model_choice}\n"
        f"Device:  {device_text}\n"
        f"Seed:    {selected_seed}\n"
        f"Steps:   {int(steps)}\n"
        f"CFG:     {float(guidance_scale):.1f}\n"
        f"Size:    {int(size)}×{int(size)}\n"
        f"Saved:   {output_path.name}"
    )
    return image, info


def generate_comparison(prompt, steps, guidance_scale, seed, size, selected_models):
    if not selected_models:
        return [], "Hãy chọn ít nhất một checkpoint."
    if not prompt or not prompt.strip():
        return [], "Vui lòng nhập prompt."

    selected_seed = normalize_seed(seed)
    gallery_items = []
    for model_choice in sorted(selected_models, key=checkpoint_sort_key):
        image, _ = generate_one(prompt, steps, guidance_scale, selected_seed, size, model_choice)
        if image is not None:
            gallery_items.append((image, model_choice))

    info = (
        f"Seed:    {selected_seed}\n"
        f"Steps:   {int(steps)}\n"
        f"CFG:     {float(guidance_scale):.1f}\n"
        f"Size:    {int(size)}×{int(size)}\n"
        f"Models:  {len(gallery_items)}\n"
        f"Order:   base → checkpoint tăng dần"
    )
    return gallery_items, info


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Bài 2 – Stable Diffusion Text-to-Image") as demo:

    gr.Markdown(
        """
# Bài 2 – Stable Diffusion Text-to-Image
Sinh ảnh biển báo giao thông Việt Nam từ mô tả văn bản.
Model base: **Stable Diffusion v1.5** · LoRA fine-tuned trên dataset biển báo VQA.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Generate single ─────────────────────────────────────────
        with gr.TabItem("🖼 Generate"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    model_dd = gr.Dropdown(
                        choices=model_choices(),
                        value=next(
                            (c for c in model_choices() if "best visual" in c),
                            model_choices()[0],
                        ),
                        label="Model / LoRA checkpoint",
                    )
                    prompt_tb = gr.Textbox(
                        label="Prompt",
                        value=REALISTIC_PROMPTS[0],
                        lines=4,
                        elem_classes=["prompt-box"],
                        placeholder="Mô tả cảnh ảnh muốn sinh...",
                    )
                    with gr.Row():
                        steps_sl = gr.Slider(10, 50, value=30, step=1, label="Steps")
                        cfg_sl = gr.Slider(1.0, 15.0, value=6.5, step=0.5, label="CFG scale")
                    with gr.Row():
                        seed_nb = gr.Number(value=-1, precision=0, label="Seed  (−1 = random)")
                        size_dd = gr.Dropdown(choices=[384, 512], value=512, label="Size")
                    gen_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")
                    info_tb = gr.Textbox(label="Info", lines=7, interactive=False)

                with gr.Column(scale=2, min_width=512):
                    output_img = gr.Image(
                        label="Generated image",
                        type="pil",
                        height=600,
                    )

            gr.Markdown("### Prompt gợi ý")
            gr.Examples(
                examples=[[p] for p in REALISTIC_PROMPTS],
                inputs=[prompt_tb],
                label="",
            )

            gen_btn.click(
                fn=generate_one,
                inputs=[prompt_tb, steps_sl, cfg_sl, seed_nb, size_dd, model_dd],
                outputs=[output_img, info_tb],
            )

        # ── Tab 2: Checkpoint comparison ───────────────────────────────────
        with gr.TabItem("📊 Comparison grid"):
            gr.Markdown(
                "Sinh ảnh từ **cùng prompt và seed** qua nhiều checkpoint để thấy sự thay đổi theo quá trình train.  \n"
                "Click vào ảnh để phóng to. Cuộn ngang để xem thêm."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    cmp_prompt = gr.Textbox(
                        label="Prompt",
                        value=REALISTIC_PROMPTS[0],
                        lines=4,
                        elem_classes=["prompt-box"],
                        placeholder="Mô tả cảnh ảnh muốn sinh...",
                    )
                    with gr.Row():
                        cmp_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                        cmp_cfg = gr.Slider(1.0, 15.0, value=6.5, step=0.5, label="CFG scale")
                    with gr.Row():
                        cmp_seed = gr.Number(value=1001, precision=0, label="Seed")
                        cmp_size = gr.Dropdown(choices=[384, 512], value=512, label="Size")
                    cmp_checks = gr.CheckboxGroup(
                        choices=model_choices(),
                        value=comparison_defaults(),
                        label="Checkpoints để so sánh",
                    )
                    cmp_btn = gr.Button("Generate comparison", variant="primary", elem_id="cmp-btn")
                    cmp_info = gr.Textbox(label="Info", lines=6, interactive=False)

                with gr.Column(scale=3, min_width=600):
                    gallery = gr.Gallery(
                        label="Kết quả theo từng checkpoint (base → 12000)",
                        columns=3,
                        rows=3,
                        height=700,
                        object_fit="contain",
                        preview=True,
                    )

            cmp_btn.click(
                fn=generate_comparison,
                inputs=[cmp_prompt, cmp_steps, cmp_cfg, cmp_seed, cmp_size, cmp_checks],
                outputs=[gallery, cmp_info],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
    )
