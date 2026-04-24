import torch


class VQACollator:
    def __call__(self, batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "decoder_input_ids": torch.stack([b["decoder_input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }
