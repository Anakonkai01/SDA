from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfigA:
    image_encoder: str = "openai/clip-vit-base-patch16"
    text_encoder: str = "vinai/phobert-base"
    co_attention_layers: int = 2
    co_attention_heads: int = 8
    dim: int = 768
    clip_dim: int = 512
    dropout: float = 0.1
    lstm_hidden: int = 768
    lstm_layers: int = 2
    transformer_layers: int = 2
    transformer_ffn: int = 3072
    vocab_size: int = 64000


@dataclass
class TrainingConfigA:
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 30
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    scheduler: str = "cosine"
    mixed_precision: bool = True
    phase1_epochs: int = 15
    phase1_lr: float = 3e-4
    phase2_epochs: int = 15
    phase2_lr: float = 3e-5


@dataclass
class DataConfigA:
    max_question_length: int = 64
    max_answer_length: int = 20
    image_size: int = 224
    num_workers: int = 4


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class ModelConfigB:
    model_name: str = "Qwen/Qwen-VL-Chat"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class TrainingConfigB:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    epochs: int = 10
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_clip: float = 1.0
    scheduler: str = "cosine"


@dataclass
class DataConfigB:
    max_length: int = 256
    image_size: int = 448
    num_workers: int = 4


@dataclass
class ConfigA:
    model: ModelConfigA = field(default_factory=ModelConfigA)
    training: TrainingConfigA = field(default_factory=TrainingConfigA)
    data: DataConfigA = field(default_factory=DataConfigA)
    vqa_data_path: str = "data/vqa/vqa_template.json"
    checkpoint_dir: str = "checkpoints"


@dataclass
class ConfigB:
    model: ModelConfigB = field(default_factory=ModelConfigB)
    training: TrainingConfigB = field(default_factory=TrainingConfigB)
    data: DataConfigB = field(default_factory=DataConfigB)
    vqa_data_path: str = "data/vqa/vqa_template.json"
    checkpoint_dir: str = "checkpoints"
