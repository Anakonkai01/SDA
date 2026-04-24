import os
import sys
import torch
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import ConfigA, ConfigB
from models.model_a import VQAModelA
from models.model_b import VQAModelB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_models = {}


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
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        clip_processor = CLIPProcessor.from_pretrained(config.model.image_encoder)
        phobert_tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder)
        loaded_models[key] = (model, clip_processor, phobert_tokenizer, config)
    return loaded_models[key]


def load_model_b(lora_path=None):
    key = "b_lora" if lora_path else "b_zeroshot"
    if key not in loaded_models:
        config = ConfigB()
        if lora_path:
            model_b = VQAModelB(model_name=config.model.model_name)
            model_b.load_lora(lora_path)
        else:
            model_b = VQAModelB(model_name=config.model.model_name)
        loaded_models[key] = model_b
    return loaded_models[key]


def predict(image, question, model_choice):
    if image is None or not question.strip():
        return "Vui long cung cap anh va cau hoi."

    if model_choice == "A1 (LSTM)":
        ckpt = "checkpoints/model_a1/best.pt"
        if not os.path.exists(ckpt):
            return "Chua co checkpoint A1. Hay train truoc."
        model, clip_proc, tokenizer, config = load_model_a("lstm", ckpt)
        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = clip_proc(
            images=pil_image, return_tensors="pt",
        ).pixel_values.to(device)
        q_enc = tokenizer(
            question, max_length=config.data.max_question_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        result = model.generate(
            pixel_values,
            q_enc.input_ids.to(device),
            q_enc.attention_mask.to(device),
        )
        return result[0]

    elif model_choice == "A2 (Transformer)":
        ckpt = "checkpoints/model_a2/best.pt"
        if not os.path.exists(ckpt):
            return "Chua co checkpoint A2. Hay train truoc."
        model, clip_proc, tokenizer, config = load_model_a("transformer", ckpt)
        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = clip_proc(
            images=pil_image, return_tensors="pt",
        ).pixel_values.to(device)
        q_enc = tokenizer(
            question, max_length=config.data.max_question_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        result = model.generate(
            pixel_values,
            q_enc.input_ids.to(device),
            q_enc.attention_mask.to(device),
        )
        return result[0]

    elif model_choice == "B1 (Zero-shot)":
        model_b = load_model_b(lora_path=None)
        temp_path = "/tmp/vqa_demo_input.png"
        Image.fromarray(image).convert("RGB").save(temp_path)
        return model_b.inference(temp_path, question)

    elif model_choice == "B2 (LoRA)":
        lora_path = "checkpoints/model_b2/best_lora"
        if not os.path.exists(lora_path):
            return "Chua co LoRA adapter B2. Hay train truoc."
        model_b = load_model_b(lora_path=lora_path)
        temp_path = "/tmp/vqa_demo_input.png"
        Image.fromarray(image).convert("RGB").save(temp_path)
        return model_b.inference(temp_path, question)

    return "Model khong hop le."


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Anh duong pho"),
        gr.Textbox(label="Cau hoi (tieng Viet)", placeholder="Bien bao ben trai la gi?"),
        gr.Dropdown(
            choices=["A1 (LSTM)", "A2 (Transformer)", "B1 (Zero-shot)", "B2 (LoRA)"],
            value="A1 (LSTM)",
            label="Chon model",
        ),
    ],
    outputs=gr.Textbox(label="Cau tra loi"),
    title="VQA Bien Bao Giao Thong Viet Nam",
    description="He thong tra loi cau hoi ve bien bao giao thong trong anh duong pho Viet Nam.",
)

if __name__ == "__main__":
    demo.launch(share=False)
