import json
import random
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_a import VQAModelA
from models.model_b import VQAModelB
from train.config import ConfigA, ConfigB

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A1_CKPT      = "checkpoints_v8_a_50k/model_a1/best.pt"
A2_CKPT      = "checkpoints_v8_a_50k/model_a2/best.pt"
B2_SFT_PATH  = "checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora"
B2_DPO_PATH  = "checkpoints_b2_dpo/best_lora"
QWEN_MAX_PIXELS = 501760

TEST_ANNOTATIONS = ROOT / "data" / "processed" / "annotations" / "test.jsonl"
EVAL_DIR         = ROOT / "data" / "human_eval"
RESULTS_FILE     = EVAL_DIR / "results.jsonl"
SCOREBOARD_FILE  = EVAL_DIR / "scoreboard.json"

MODELS = ["A1", "A2", "B1", "B2_SFT", "B2_DPO"]
MODEL_LABELS = {
    "A1":     "A1 (LSTM)",
    "A2":     "A2 (Transformer)",
    "B1":     "B1 (Zero-shot)",
    "B2_SFT": "B2 (SFT)",
    "B2_DPO": "B2-DPO (100p)",
}

loaded_models: dict = {}
sample_records: list = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_path(image_path: str) -> str:
    for base in [ROOT, ROOT / "data" / "processed"]:
        p = base / image_path
        if p.exists():
            return str(p)
    return str(ROOT / image_path)


def q_label(i: int, q: str) -> str:
    s = q[:65] + ("…" if len(q) > 65 else "")
    return f"Q{i+1}: {s}"


def parse_correct_indices(ck_values) -> set:
    idxs = set()
    for v in (ck_values or []):
        try:
            idxs.add(int(v.split(":")[0][1:]) - 1)  # "Q3: ..." → 2
        except Exception:
            pass
    return idxs


# ── Scoreboard ────────────────────────────────────────────────────────────────

def load_scoreboard() -> dict:
    if SCOREBOARD_FILE.exists():
        return json.loads(SCOREBOARD_FILE.read_text(encoding="utf-8"))
    return {m: {"correct": 0, "total": 0} for m in MODELS}


def save_scoreboard(sb: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SCOREBOARD_FILE.write_text(
        json.dumps(sb, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def make_scoreboard_df(sb: dict | None = None) -> pd.DataFrame:
    if sb is None:
        sb = load_scoreboard()
    rows = []
    for m in MODELS:
        c, t = sb[m]["correct"], sb[m]["total"]
        rows.append([MODEL_LABELS[m], c, t, f"{c/t*100:.1f}%" if t else "—"])
    return pd.DataFrame(rows, columns=["Model", "Đúng", "Tổng", "Accuracy"])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_a(decoder_type: str, ckpt_path: str):
    key = f"a_{decoder_type}"
    if key not in loaded_models:
        config = ConfigA()
        model = VQAModelA(
            decoder_type=decoder_type,
            vocab_size=config.model.vocab_size,
            dim=config.model.dim,
            clip_dim=config.model.clip_dim,
        ).to(device)
        ckpt = torch.load(ROOT / ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        clip_proc = CLIPProcessor.from_pretrained(config.model.image_encoder)
        tok = AutoTokenizer.from_pretrained(config.model.text_encoder)
        loaded_models[key] = (model, clip_proc, tok, config)
    return loaded_models[key]


def load_model_b(lora_path: str | None = None):
    key = f"b:{lora_path}"
    if key not in loaded_models:
        config = ConfigB()
        mb = VQAModelB(
            model_name=config.model.qwen_model_name,
            backend="qwen25",
            load_in_4bit=True,
            max_pixels=QWEN_MAX_PIXELS,
        )
        if lora_path:
            mb.load_lora(ROOT / lora_path)
        loaded_models[key] = mb
    return loaded_models[key]


def infer_a(decoder_type: str, ckpt_path: str, pil: Image.Image, question: str) -> str:
    model, clip_proc, tok, config = load_model_a(decoder_type, ckpt_path)
    pv = clip_proc(images=pil, return_tensors="pt").pixel_values.to(device)
    enc = tok(
        question,
        max_length=config.data.max_question_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return model.generate(pv, enc.input_ids.to(device), enc.attention_mask.to(device))[0]


def infer_b(lora_path: str | None, pil: Image.Image, question: str) -> str:
    mb = load_model_b(lora_path)
    tmp = "/tmp/_human_eval_img.png"
    pil.save(tmp)
    return mb.inference(tmp, question)


# ── Test sample loading ───────────────────────────────────────────────────────

def load_test_samples() -> list:
    seen, records = set(), []
    if not TEST_ANNOTATIONS.exists():
        return records
    with TEST_ANNOTATIONS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            iid, ipath = row.get("image_id", ""), row.get("image_path", "")
            if not iid or iid in seen or not ipath:
                continue
            if not Path(resolve_path(ipath)).exists():
                continue
            seen.add(iid)
            records.append({"image_id": iid, "image_path": ipath})
    return records


def make_gallery_items(indices: list) -> list:
    return [
        (resolve_path(sample_records[i]["image_path"]), sample_records[i]["image_id"])
        for i in indices
    ]


# ── Gradio callbacks ──────────────────────────────────────────────────────────

def shuffle_gallery():
    indices = random.sample(range(len(sample_records)), min(8, len(sample_records)))
    return make_gallery_items(indices), indices


def select_image(indices: list, evt: gr.SelectData):
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    sample = sample_records[indices[idx]]
    img_path = resolve_path(sample["image_path"])
    pil = Image.open(img_path).convert("RGB")
    return pil, sample["image_id"], img_path, sample["image_id"]


def _empty_ck():
    return gr.update(choices=[], value=[])


def run_all_models(img_path: str, questions_text: str):
    if not img_path:
        return [], "⚠ Chưa chọn ảnh.", None, _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck()

    questions = [q.strip() for q in (questions_text or "").split("\n") if q.strip()]
    if not questions:
        return [], "⚠ Chưa nhập câu hỏi.", None, _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck()

    pil = Image.open(img_path).convert("RGB")
    results = []

    for q in questions:
        answers = {}
        answers["A1"] = (
            infer_a("lstm", A1_CKPT, pil, q)
            if (ROOT / A1_CKPT).exists() else "[checkpoint missing]"
        )
        answers["A2"] = (
            infer_a("transformer", A2_CKPT, pil, q)
            if (ROOT / A2_CKPT).exists() else "[checkpoint missing]"
        )
        answers["B1"] = infer_b(None, pil, q)
        answers["B2_SFT"] = (
            infer_b(B2_SFT_PATH, pil, q)
            if (ROOT / B2_SFT_PATH).exists() else "[checkpoint missing]"
        )
        answers["B2_DPO"] = (
            infer_b(B2_DPO_PATH, pil, q)
            if (ROOT / B2_DPO_PATH).exists() else "[checkpoint missing]"
        )
        results.append({"question": q, "answers": answers})

    df = pd.DataFrame(
        [
            [f"Q{i+1}", r["question"],
             r["answers"]["A1"], r["answers"]["A2"], r["answers"]["B1"],
             r["answers"]["B2_SFT"], r["answers"]["B2_DPO"]]
            for i, r in enumerate(results)
        ],
        columns=["#", "Câu hỏi", "A1", "A2", "B1", "B2-SFT", "B2-DPO"],
    )

    choices = [q_label(i, r["question"]) for i, r in enumerate(results)]
    status  = f"✅ Chạy xong {len(questions)} câu × 5 model. Đánh dấu đúng/sai bên dưới rồi bấm Lưu."
    return (results, status, df,
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]))


def save_and_score(image_id, img_path, results,
                   ck_a1, ck_a2, ck_b1, ck_b2sft, ck_b2dpo):
    if not results:
        return ("⚠ Chưa có kết quả.", make_scoreboard_df(),
                _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck())

    correct_map = {
        "A1":     parse_correct_indices(ck_a1),
        "A2":     parse_correct_indices(ck_a2),
        "B1":     parse_correct_indices(ck_b1),
        "B2_SFT": parse_correct_indices(ck_b2sft),
        "B2_DPO": parse_correct_indices(ck_b2dpo),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sb = load_scoreboard()

    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        for i, r in enumerate(results):
            correct = [m for m in MODELS if i in correct_map[m]]
            wrong   = [m for m in MODELS if m not in correct]
            entry = {
                "timestamp":    datetime.now().isoformat(),
                "image_id":     image_id,
                "image_path":   img_path,
                "question_idx": i,
                "question":     r["question"],
                "answers":      r["answers"],
                "correct":      correct,
                "wrong":        wrong,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for m in MODELS:
                sb[m]["total"] += 1
                if m in correct:
                    sb[m]["correct"] += 1

    save_scoreboard(sb)
    return (f"✅ Đã lưu {len(results)} câu vào {RESULTS_FILE.name}. Scoreboard cập nhật.",
            make_scoreboard_df(sb),
            _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck(), _empty_ck())


# ── Bootstrap ─────────────────────────────────────────────────────────────────

sample_records = load_test_samples()
_init_indices  = random.sample(range(len(sample_records)), min(8, len(sample_records)))
_init_gallery  = make_gallery_items(_init_indices)
_init_sample   = sample_records[_init_indices[0]]
_init_img_path = resolve_path(_init_sample["image_path"])
_init_img_id   = _init_sample["image_id"]
_init_pil      = Image.open(_init_img_path).convert("RGB")


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Human Eval — VQA") as demo:
    gr.Markdown(
        "# Human Evaluation — VQA Biển Báo Giao Thông\n"
        "**Quy trình:** Chọn ảnh → Nhập câu hỏi → Run → Tick đúng/sai → Lưu"
    )

    # ── Scoreboard ─────────────────────────────────────────────────────────
    gr.Markdown("## Scoreboard")
    scoreboard_tbl = gr.Dataframe(value=make_scoreboard_df(), interactive=False, wrap=True)

    # ── State ──────────────────────────────────────────────────────────────
    gallery_state  = gr.State(_init_indices)
    sel_img_path   = gr.State(_init_img_path)
    sel_img_id     = gr.State(_init_img_id)
    infer_results  = gr.State([])

    # ── Gallery ────────────────────────────────────────────────────────────
    gr.Markdown("## Chọn ảnh từ tập test")
    shuffle_btn = gr.Button("🔀 Shuffle ảnh", size="sm")
    gallery = gr.Gallery(
        value=_init_gallery,
        label="Click vào ảnh để chọn",
        columns=4, rows=2, height=220,
        object_fit="cover",
        allow_preview=False,
    )

    # ── Work area ──────────────────────────────────────────────────────────
    gr.Markdown("## Câu hỏi & Kết quả")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            selected_img = gr.Image(
                value=_init_pil,
                label="Ảnh đang chọn",
                type="pil",
                height=340,
                interactive=False,
            )
            img_id_box = gr.Textbox(
                value=_init_img_id,
                label="Image ID",
                interactive=False,
            )
        with gr.Column(scale=2):
            questions_tb = gr.Textbox(
                label="Nhập câu hỏi — mỗi dòng 1 câu",
                placeholder=(
                    "Biển báo màu gì?\n"
                    "Có bao nhiêu biển báo trong ảnh?\n"
                    "Biển báo nằm ở đâu?"
                ),
                lines=6,
            )
            run_btn   = gr.Button("▶ Run tất cả 5 model", variant="primary")
            status_tb = gr.Textbox(label="Trạng thái", interactive=False, lines=1)

    results_tbl = gr.Dataframe(
        label="Kết quả inference",
        interactive=False,
        wrap=True,
    )

    # ── Mark correct ───────────────────────────────────────────────────────
    gr.Markdown(
        "## Đánh dấu đúng/sai\n"
        "_Tick vào những câu mà model đó trả lời **đúng**. Không tick = sai._"
    )
    with gr.Row():
        ck_a1    = gr.CheckboxGroup(label="A1 (LSTM) — đúng",        choices=[], scale=1)
        ck_a2    = gr.CheckboxGroup(label="A2 (Transformer) — đúng", choices=[], scale=1)
    with gr.Row():
        ck_b1    = gr.CheckboxGroup(label="B1 (Zero-shot) — đúng",   choices=[], scale=1)
        ck_b2sft = gr.CheckboxGroup(label="B2 (SFT) — đúng",         choices=[], scale=1)
        ck_b2dpo = gr.CheckboxGroup(label="B2-DPO (100p) — đúng",    choices=[], scale=1)

    with gr.Row():
        save_btn    = gr.Button("💾 Lưu & Cập nhật scoreboard", variant="primary")
        save_status = gr.Textbox(label="", interactive=False, lines=1, scale=3)

    # ── Wiring ─────────────────────────────────────────────────────────────

    shuffle_btn.click(
        fn=shuffle_gallery,
        outputs=[gallery, gallery_state],
    )

    gallery.select(
        fn=select_image,
        inputs=[gallery_state],
        outputs=[selected_img, img_id_box, sel_img_path, sel_img_id],
    )

    run_btn.click(
        fn=run_all_models,
        inputs=[sel_img_path, questions_tb],
        outputs=[infer_results, status_tb, results_tbl,
                 ck_a1, ck_a2, ck_b1, ck_b2sft, ck_b2dpo],
    )

    save_btn.click(
        fn=save_and_score,
        inputs=[sel_img_id, sel_img_path, infer_results,
                ck_a1, ck_a2, ck_b1, ck_b2sft, ck_b2dpo],
        outputs=[save_status, scoreboard_tbl,
                 ck_a1, ck_a2, ck_b1, ck_b2sft, ck_b2dpo],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
