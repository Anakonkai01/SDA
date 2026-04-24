import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class VQAModelB:
    def __init__(self, model_name="Qwen/Qwen-VL-Chat", lora_config=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )

        if lora_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            self.is_lora = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self.model.eval()
            self.is_lora = False

    @torch.no_grad()
    def inference(self, image_path, question):
        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])
        response, _ = self.model.chat(
            self.tokenizer, query=query, history=None,
        )
        return response

    def prepare_training_input(self, image_path, question, answer):
        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])
        prompt = query + answer + self.tokenizer.eos_token
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256,
        )
        labels = encoded.input_ids.clone()

        query_only = self.tokenizer(query, return_tensors="pt")
        query_len = query_only.input_ids.size(1)
        labels[:, :query_len] = -100

        return {
            "input_ids": encoded.input_ids.squeeze(),
            "attention_mask": encoded.attention_mask.squeeze(),
            "labels": labels.squeeze(),
        }

    def save_lora(self, path):
        if self.is_lora:
            self.model.save_pretrained(path)
            print(f"LoRA adapter saved to {path}")

    def load_lora(self, path):
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, path)
        self.model.eval()
        self.is_lora = True
