import json
import os
import random
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_a import VQAModelA
from models.model_b import VQAModelB
from train.config import ConfigA, ConfigB

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A1_CHECKPOINT = "checkpoints_v8_a_50k/model_a1/best.pt"
A2_CHECKPOINT = "checkpoints_v8_a_50k/model_a2/best.pt"
B2_LORA_PATH = "checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora"
DPO_BEST_PATH = "checkpoints_b2_dpo/best_lora"
DPO_COLLAPSE_PATH = "checkpoints_b2_dpo_full/best_lora"
QWEN_MAX_PIXELS = 501760
TEST_ANNOTATIONS = ROOT / "data" / "processed" / "annotations" / "test.jsonl"

loaded_models = {}
sample_records = []


def rel_exists(path):
    return (ROOT / path).exists()


def resolve_image_path(image_path):
    path = ROOT / image_path
    if path.exists():
        return path
    processed_path = ROOT / "data" / "processed" / image_path
    if processed_path.exists():
        return processed_path
    return path


def load_test_samples(questions_per_image=8):
    by_image = {}
    if not TEST_ANNOTATIONS.exists():
        return []
    with TEST_ANNOTATIONS.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_id = row.get("image_id", "")
            image_path = row.get("image_path", "")
            if not image_id or not image_path:
                continue
            if not resolve_image_path(image_path).exists():
                continue
            item = by_image.setdefault(
                image_id,
                {"image_id": image_id, "image_path": image_path, "questions": []},
            )
            if len(item["questions"]) < questions_per_image:
                question = row.get("question", "")
                if question:
                    item["questions"].append(question)
    return [sample for sample in by_image.values() if sample["questions"]]


def make_gallery(indices):
    items = []
    for idx in indices:
        sample = sample_records[idx]
        items.append((str(resolve_image_path(sample["image_path"])), sample["image_id"]))
    return items


def initial_indices():
    return list(range(min(4, len(sample_records))))


def get_sample_by_position(indices, selected):
    if not indices:
        return None, gr.update(choices=[], value=None), ""
    selected = max(0, min(int(selected or 0), len(indices) - 1))
    sample = sample_records[indices[selected]]
    image = Image.open(resolve_image_path(sample["image_path"])).convert("RGB")
    questions = sample["questions"]
    first_question = questions[0] if questions else ""
    return image, gr.update(choices=questions, value=first_question), first_question


def shuffle_gallery():
    if len(sample_records) <= 4:
        indices = initial_indices()
    else:
        indices = random.sample(range(len(sample_records)), 4)
    image_value, question_update, question_value = get_sample_by_position(indices, 0)
    return make_gallery(indices), indices, image_value, question_update, question_value


def select_sample_image(indices, evt: gr.SelectData):
    selected = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return get_sample_by_position(indices, selected)


def select_question(question_choice):
    return question_choice or ""


def load_model_a(decoder_type, checkpoint_path):
    key = f"a_{decoder_type}"
    if key not in loaded_models:
        config = ConfigA()
        model = VQAModelA(
            decoder_type=decoder_type,
            vocab_size=config.model.vocab_size,
            dim=config.model.dim,
            clip_dim=config.model.clip_dim,
        ).to(device)
        ckpt = torch.load(ROOT / checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        clip_processor = CLIPProcessor.from_pretrained(config.model.image_encoder)
        phobert_tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder)
        loaded_models[key] = (model, clip_processor, phobert_tokenizer, config)
    return loaded_models[key]


def load_model_b(lora_path=None):
    key = f"b_lora:{lora_path}" if lora_path else "b_zeroshot"
    if key not in loaded_models:
        config = ConfigB()
        model_b = VQAModelB(
            model_name=config.model.qwen_model_name,
            backend="qwen25",
            load_in_4bit=True,
            max_pixels=QWEN_MAX_PIXELS,
        )
        if lora_path:
            model_b.load_lora(ROOT / lora_path)
        loaded_models[key] = model_b
    return loaded_models[key]


def as_pil_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def predict(image, question, model_choice):
    if image is None or not question.strip():
        return "Vui lòng cung cấp ảnh và câu hỏi."

    pil_image = as_pil_image(image)
    question = question.strip()

    if model_choice == "A1 (LSTM)":
        ckpt = A1_CHECKPOINT
        if not rel_exists(ckpt):
            return f"Chưa có checkpoint A1 tại {ckpt}."
        model, clip_proc, tokenizer, config = load_model_a("lstm", ckpt)
        pixel_values = clip_proc(images=pil_image, return_tensors="pt").pixel_values.to(device)
        q_enc = tokenizer(
            question,
            max_length=config.data.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result = model.generate(
            pixel_values,
            q_enc.input_ids.to(device),
            q_enc.attention_mask.to(device),
        )
        return result[0]

    if model_choice == "A2 (Transformer)":
        ckpt = A2_CHECKPOINT
        if not rel_exists(ckpt):
            return f"Chưa có checkpoint A2 tại {ckpt}."
        model, clip_proc, tokenizer, config = load_model_a("transformer", ckpt)
        pixel_values = clip_proc(images=pil_image, return_tensors="pt").pixel_values.to(device)
        q_enc = tokenizer(
            question,
            max_length=config.data.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result = model.generate(
            pixel_values,
            q_enc.input_ids.to(device),
            q_enc.attention_mask.to(device),
        )
        return result[0]

    if model_choice == "B1 (Zero-shot)":
        model_b = load_model_b(lora_path=None)
        temp_path = "/tmp/vqa_demo_input.png"
        pil_image.save(temp_path)
        return model_b.inference(temp_path, question)

    if model_choice == "B2 (LoRA)":
        lora_path = B2_LORA_PATH
        if not rel_exists(lora_path):
            return f"Chưa có LoRA adapter B2 tại {lora_path}."
        model_b = load_model_b(lora_path=lora_path)
        temp_path = "/tmp/vqa_demo_input.png"
        pil_image.save(temp_path)
        return model_b.inference(temp_path, question)

    if model_choice == "B2-DPO (100p best)":
        lora_path = DPO_BEST_PATH
        if not rel_exists(lora_path):
            return f"Chưa có DPO adapter tại {lora_path}."
        model_b = load_model_b(lora_path=lora_path)
        temp_path = "/tmp/vqa_demo_input.png"
        pil_image.save(temp_path)
        return model_b.inference(temp_path, question)

    if model_choice == "B2-DPO (3146p collapse)":
        lora_path = DPO_COLLAPSE_PATH
        if not rel_exists(lora_path):
            return f"Chưa có DPO adapter tại {lora_path}."
        model_b = load_model_b(lora_path=lora_path)
        temp_path = "/tmp/vqa_demo_input.png"
        pil_image.save(temp_path)
        return model_b.inference(temp_path, question)

    return "Model không hợp lệ."


sample_records = load_test_samples()
indices0 = initial_indices()
gallery0 = make_gallery(indices0)
image0 = None
question_choices0 = []
question0 = ""
if indices0:
    first_sample = sample_records[indices0[0]]
    image0 = Image.open(resolve_image_path(first_sample["image_path"])).convert("RGB")
    question_choices0 = first_sample["questions"]
    question0 = question_choices0[0] if question_choices0 else ""

with gr.Blocks(title="VQA Biển Báo Giao Thông Việt Nam") as demo:
    gr.Markdown(
        "# VQA Biển Báo Giao Thông Việt Nam\n"
        "Chọn trực tiếp ảnh test bên dưới, bấm shuffle để đổi ảnh, hoặc upload/paste ảnh của bạn. "
        "Bạn có thể chọn câu hỏi gợi ý hoặc tự nhập câu hỏi riêng."
    )
    gallery_indices = gr.State(indices0)

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=[
                    "A1 (LSTM)",
                    "A2 (Transformer)",
                    "B1 (Zero-shot)",
                    "B2 (LoRA)",
                    "B2-DPO (100p best)",
                    "B2-DPO (3146p collapse)",
                ],
                value="A1 (LSTM)",
                label="Chọn model",
            )
            shuffle_button = gr.Button("Shuffle 4 ảnh test")
            sample_gallery = gr.Gallery(
                value=gallery0,
                label="Chọn 1 ảnh test mẫu",
                columns=4,
                rows=1,
                height=180,
                object_fit="cover",
                allow_preview=False,
            )
            suggested_question = gr.Radio(
                choices=question_choices0,
                value=question0,
                label="Câu hỏi gợi ý cho ảnh đang chọn",
            )
            question = gr.Textbox(
                label="Nhập câu hỏi của bạn",
                value=question0,
                placeholder="Ví dụ: Trong ảnh có biển cấm rẽ trái không?",
                lines=2,
            )
            run_button = gr.Button("Trả lời", variant="primary")
        with gr.Column(scale=1):
            image = gr.Image(
                label="Ảnh đang dùng để hỏi (có thể upload hoặc paste Ctrl+V)",
                value=image0,
                sources=["upload", "clipboard"],
                type="numpy",
                image_mode="RGB",
                height=380,
            )
            output = gr.Textbox(label="Câu trả lời", lines=3)

    shuffle_button.click(
        fn=shuffle_gallery,
        inputs=None,
        outputs=[sample_gallery, gallery_indices, image, suggested_question, question],
    )
    sample_gallery.select(
        fn=select_sample_image,
        inputs=gallery_indices,
        outputs=[image, suggested_question, question],
    )
    suggested_question.change(
        fn=select_question,
        inputs=suggested_question,
        outputs=question,
    )
    run_button.click(
        fn=predict,
        inputs=[image, question, model_choice],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
