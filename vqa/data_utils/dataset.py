import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class VQADataset(Dataset):
    def __init__(self, samples, clip_processor, phobert_tokenizer,
                 max_q_len=64, max_a_len=20, image_size=224):
        self.samples = samples
        self.clip_processor = clip_processor
        self.phobert_tokenizer = phobert_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")

        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        q_enc = self.phobert_tokenizer(
            sample["question"],
            max_length=self.max_q_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        a_enc = self.phobert_tokenizer(
            sample["answer"],
            max_length=self.max_a_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        bos_id = self.phobert_tokenizer.bos_token_id
        pad_id = self.phobert_tokenizer.pad_token_id

        decoder_input_ids = torch.cat([
            torch.tensor([bos_id]),
            a_enc.input_ids.squeeze()[:-1],
        ])

        labels = a_enc.input_ids.squeeze().clone()
        labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids.squeeze(),
            "attention_mask": q_enc.attention_mask.squeeze(),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }


def load_vqa_data(json_path, split="train"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[split]
