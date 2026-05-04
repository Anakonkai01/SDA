from pathlib import Path

import torch
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering, BlipProcessor


BLIP_DEFAULT_LORA_TARGETS = ("query", "key", "value")
QWEN_DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PALIGEMMA_DEFAULT_LORA_TARGETS = QWEN_DEFAULT_LORA_TARGETS


class VQAModelB:
    """Pretrained multimodal VQA model for B1/B2.

    The default backend remains BLIP for backwards-compatible smoke runs.
    Use backend="qwen25" for the stronger Qwen2.5-VL path, or
    backend="paligemma2" for the PaliGemma 2 pilot path.
    """

    def __init__(self, model_name=None, lora_config=None, device=None,
                 backend="blip", load_in_4bit=False, min_pixels=None,
                 max_pixels=None):
        self.backend = backend.lower()
        if self.backend not in ("blip", "qwen25", "paligemma2"):
            raise ValueError(f"Unsupported B backend: {backend}")

        if model_name is None:
            if self.backend == "qwen25":
                model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            elif self.backend == "paligemma2":
                model_name = "google/paligemma2-3b-pt-448"
            else:
                model_name = "Salesforce/blip-vqa-base"
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = self._load_processor()
        self.model = self._load_base_model()
        self.is_lora = False

        if lora_config is not None:
            self._attach_lora(lora_config)
            self.is_lora = True
        else:
            self.model.eval()

        if not self._uses_device_map():
            self.model.to(self.device)

    def _handle_hf_access_error(self, exc):
        raise RuntimeError(
            f"Cannot access Hugging Face model '{self.model_name}'. "
            "If this is a gated/private model, request access on its Hugging Face "
            "model page and authenticate in this environment with `huggingface-cli "
            "login` or the `HF_TOKEN` environment variable. For PaliGemma 2, the "
            "Google model repos are gated, so this is expected until access is "
            "approved for your HF account."
        ) from exc

    def _load_processor(self):
        if self.backend == "blip":
            try:
                return BlipProcessor.from_pretrained(self.model_name)
            except (GatedRepoError, RepositoryNotFoundError, OSError) as exc:
                self._handle_hf_access_error(exc)

        kwargs = {}
        if self.backend == "qwen25":
            if self.min_pixels is not None:
                kwargs["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                kwargs["max_pixels"] = self.max_pixels
        try:
            processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        except (GatedRepoError, RepositoryNotFoundError, OSError) as exc:
            self._handle_hf_access_error(exc)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
        return processor

    def _load_base_model(self):
        if self.backend == "blip":
            try:
                return BlipForQuestionAnswering.from_pretrained(self.model_name)
            except (GatedRepoError, RepositoryNotFoundError, OSError) as exc:
                self._handle_hf_access_error(exc)

        from transformers import BitsAndBytesConfig

        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.load_in_4bit:
            model_kwargs.pop("torch_dtype", None)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        if self.backend == "qwen25":
            from transformers import Qwen2_5_VLForConditionalGeneration

            try:
                return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
            except (GatedRepoError, RepositoryNotFoundError, OSError) as exc:
                self._handle_hf_access_error(exc)

        from transformers import PaliGemmaForConditionalGeneration

        try:
            return PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        except (GatedRepoError, RepositoryNotFoundError, OSError) as exc:
            self._handle_hf_access_error(exc)

    def _uses_device_map(self):
        return self.backend in ("qwen25", "paligemma2") and self.load_in_4bit

    def _attach_lora(self, lora_config):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        target_modules = list(lora_config.target_modules)
        if (
            self.backend == "qwen25"
            and tuple(target_modules) == BLIP_DEFAULT_LORA_TARGETS
        ):
            target_modules = list(QWEN_DEFAULT_LORA_TARGETS)
        if (
            self.backend == "paligemma2"
            and tuple(target_modules) == BLIP_DEFAULT_LORA_TARGETS
        ):
            target_modules = list(PALIGEMMA_DEFAULT_LORA_TARGETS)

        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
        )
        self.model = get_peft_model(self.model, peft_config)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model.print_trainable_parameters()

    @staticmethod
    def _load_image(image_path):
        return Image.open(Path(image_path)).convert("RGB")

    def _qwen_prompt(self, question):
        question = str(question).strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {
                        "type": "text",
                        "text": (
                            "Trả lời ngắn gọn bằng tiếng Việt. "
                            "Chỉ đưa đáp án, không giải thích.\n"
                            f"Câu hỏi: {question}"
                        ),
                    },
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _paligemma_prompt(question):
        question = str(question).strip()
        return f"<image>answer vi {question}\n"

    def _to_device(self, inputs):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

    @torch.no_grad()
    def inference(self, image_path, question, max_new_tokens=20):
        return self.inference_batch(
            [image_path],
            [question],
            max_new_tokens=max_new_tokens,
        )[0]

    @torch.no_grad()
    def inference_batch(self, image_paths, questions, max_new_tokens=20):
        images = [self._load_image(image_path) for image_path in image_paths]
        if self.backend == "qwen25":
            old_padding_side = self.processor.tokenizer.padding_side
            try:
                self.processor.tokenizer.padding_side = "left"
                prompts = [self._qwen_prompt(question) for question in questions]
                inputs = self.processor(
                    text=prompts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
            finally:
                self.processor.tokenizer.padding_side = old_padding_side
            inputs = self._to_device(inputs)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated = generated[:, inputs["input_ids"].shape[1]:]
            return [
                text.strip()
                for text in self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ]
        if self.backend == "paligemma2":
            prompts = [self._paligemma_prompt(question) for question in questions]
            inputs = self.processor(
                text=prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = self._to_device(inputs)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated = generated[:, inputs["input_ids"].shape[1]:]
            return [
                text.strip()
                for text in self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ]

        inputs = self.processor(
            images=images,
            text=list(questions),
            padding=True,
            return_tensors="pt",
        )
        inputs = self._to_device(inputs)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        return [
            text.strip()
            for text in self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def prepare_training_input(self, image_path, question, answer,
                               max_question_length=64,
                               max_answer_length=20):
        if self.backend in ("qwen25", "paligemma2"):
            raise NotImplementedError(
                f"{self.backend} training uses prepare_training_batch()."
            )

        image = self._load_image(image_path)
        inputs = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt",
        )
        answer_tokens = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=max_answer_length,
            return_tensors="pt",
        )
        decoder_input_ids = answer_tokens.input_ids
        decoder_attention_mask = answer_tokens.attention_mask
        labels = decoder_input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "decoder_input_ids": decoder_input_ids.squeeze(0),
            "decoder_attention_mask": decoder_attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def prepare_training_batch(self, samples, max_length=1024):
        if self.backend not in ("qwen25", "paligemma2"):
            raise NotImplementedError(
                "prepare_training_batch() is only used by generative B backends."
            )
        if self.backend == "paligemma2" and max_length is not None and max_length < 1024:
            raise ValueError(
                "PaliGemma 2 requires max_length >= 1024 because image tokens "
                "already occupy that context budget. Remove --max-length or use "
                "a larger value such as 1280."
            )

        images = [self._load_image(sample["image"]) for sample in samples]
        if self.backend == "qwen25":
            prompts = [self._qwen_prompt(sample["question"]) for sample in samples]
        else:
            prompts = [
                self._paligemma_prompt(sample["question"])
                for sample in samples
            ]
        eos_token = self.processor.tokenizer.eos_token or ""
        texts = [
            f"{prompt}{str(sample['answer']).strip()}{eos_token}"
            for prompt, sample in zip(prompts, samples)
        ]

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = inputs.input_ids.clone()

        for i, prompt in enumerate(prompts):
            prompt_inputs = self.processor(
                text=[prompt],
                images=[images[i]],
                padding=False,
                return_tensors="pt",
            )
            prompt_len = min(prompt_inputs.input_ids.shape[1], labels.shape[1])
            labels[i, :prompt_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch = {
            key: value
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        batch["labels"] = labels
        return batch

    def save_lora(self, path):
        if not self.is_lora:
            raise RuntimeError("Cannot save LoRA adapter from a non-LoRA model.")
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"LoRA adapter saved to {path}")

    def load_lora(self, path):
        from peft import PeftModel

        base_model = self.model
        self.model = PeftModel.from_pretrained(base_model, path)
        if not self._uses_device_map():
            self.model.to(self.device)
        self.model.eval()
        self.is_lora = True
