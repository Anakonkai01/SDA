import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


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


def _read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalize_record(record, processed_dir=None):
    sample = dict(record)

    if "image" not in sample:
        image_path = sample.get("image_path")
        if image_path is not None:
            image = Path(image_path)
            if processed_dir is not None and not image.is_absolute():
                image = processed_dir / image
            sample["image"] = str(image)

    sample.setdefault("type", sample.get("question_type", "unknown"))
    return sample


def load_vqa_data(data_path, split="train"):
    """Load legacy JSON data or new split JSONL annotations.

    Supported inputs:
    - legacy JSON file with top-level train/val/test lists
    - annotations directory containing train.jsonl/val.jsonl/test.jsonl
    - one JSONL file, optionally filtered by its split field
    """
    path = Path(data_path)

    if path.is_dir():
        jsonl_path = path / f"{split}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing split file: {jsonl_path}")
        processed_dir = path.parent
        return [_normalize_record(r, processed_dir) for r in _read_jsonl(jsonl_path)]

    if path.suffix.lower() == ".jsonl":
        processed_dir = path.parent.parent if path.parent.name.startswith("annotations") else path.parent
        records = [
            r for r in _read_jsonl(path)
            if not r.get("split") or r.get("split") == split
        ]
        return [_normalize_record(r, processed_dir) for r in records]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [_normalize_record(r) for r in data[split]]
