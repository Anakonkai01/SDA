"""
Script sinh câu hỏi VQA từ YOLO annotation + paraphrase bằng Qwen2.5 14B qua Ollama

Usage:
  # Bước 1: Sinh template
  python generate_vqa.py --mode template --data_dir data/datasets/remapped_v2 --output vqa_template.json

  # Bước 2: Paraphrase bằng LLM (cần Ollama + qwen2.5:14b đã pull)
  python generate_vqa.py --mode paraphrase --input vqa_template.json --output vqa_final.json

  # Test nhanh với 100 mẫu
  python generate_vqa.py --mode paraphrase --input vqa_template.json --output vqa_test.json --max_samples 100

  # Chạy cả 2 bước liên tiếp
  python generate_vqa.py --mode full --data_dir data/datasets/remapped_v2 --output vqa_final.json
"""

import os
import json
import yaml
import random
import argparse
import requests
import time
from pathlib import Path
from collections import defaultdict

# ============================================================
# THÔNG TIN TỪNG CLASS
# ============================================================
CLASS_INFO = {
    "DP_het_tat_ca_lenh_cam":    {"desc": "biển hết tất cả lệnh cấm",              "group": "biển hết cấm",    "action": "các lệnh cấm trước đó không còn hiệu lực",       "color": "trắng và đen"},
    "I_cho_quay_xe":             {"desc": "biển chỉ dẫn chỗ quay xe",              "group": "biển chỉ dẫn",   "action": "có thể quay đầu xe tại đây",                     "color": "xanh và trắng"},
    "I_duong_mot_chieu":         {"desc": "biển đường một chiều",                   "group": "biển chỉ dẫn",   "action": "chỉ được đi theo một chiều",                     "color": "xanh và trắng"},
    "P_cam_coi":                 {"desc": "biển cấm sử dụng còi",                   "group": "biển cấm",       "action": "không được bấm còi",                             "color": "đỏ và trắng"},
    "P_cam_dung_do_xe":          {"desc": "biển cấm dừng và đỗ xe",                 "group": "biển cấm",       "action": "không được dừng hoặc đỗ xe",                     "color": "đỏ và trắng"},
    "P_cam_nguoc_chieu":         {"desc": "biển cấm đi ngược chiều",                "group": "biển cấm",       "action": "không được đi vào vì đi ngược chiều",            "color": "đỏ và trắng"},
    "P_cam_nguoi_di_bo":         {"desc": "biển cấm người đi bộ",                   "group": "biển cấm",       "action": "người đi bộ không được đi vào",                  "color": "đỏ và trắng"},
    "P_cam_quay_dau":            {"desc": "biển cấm quay đầu xe",                   "group": "biển cấm",       "action": "không được quay đầu xe",                         "color": "đỏ và trắng"},
    "P_cam_re_ca_hai_chieu":     {"desc": "biển cấm rẽ cả hai chiều",               "group": "biển cấm",       "action": "không được rẽ trái hoặc rẽ phải",                "color": "đỏ và trắng"},
    "P_cam_re_phai":             {"desc": "biển cấm rẽ phải",                       "group": "biển cấm",       "action": "không được rẽ phải",                             "color": "đỏ và trắng"},
    "P_cam_re_trai":             {"desc": "biển cấm rẽ trái",                       "group": "biển cấm",       "action": "không được rẽ trái",                             "color": "đỏ và trắng"},
    "P_cam_vuot":                {"desc": "biển cấm vượt",                          "group": "biển cấm",       "action": "không được vượt xe khác",                        "color": "đỏ và trắng"},
    "P_cam_xe_khach":            {"desc": "biển cấm xe khách",                      "group": "biển cấm",       "action": "xe khách không được đi vào",                     "color": "đỏ và trắng"},
    "P_cam_xe_may":              {"desc": "biển cấm xe máy",                        "group": "biển cấm",       "action": "xe máy không được đi vào",                       "color": "đỏ và trắng"},
    "P_cam_xe_oto":              {"desc": "biển cấm xe ô tô",                       "group": "biển cấm",       "action": "xe ô tô không được đi vào",                      "color": "đỏ và trắng"},
    "P_cam_xe_oto_re":           {"desc": "biển cấm xe ô tô rẽ",                   "group": "biển cấm",       "action": "xe ô tô không được rẽ",                          "color": "đỏ và trắng"},
    "P_cam_xe_tai":              {"desc": "biển cấm xe tải",                        "group": "biển cấm",       "action": "xe tải không được đi vào",                       "color": "đỏ và trắng"},
    "P_han_che_chieu_cao":       {"desc": "biển hạn chế chiều cao",                 "group": "biển cấm",       "action": "chú ý giới hạn chiều cao xe",                    "color": "đỏ và trắng"},
    "P_han_che_trong_tai":       {"desc": "biển hạn chế trọng tải",                 "group": "biển cấm",       "action": "chú ý giới hạn trọng tải xe",                    "color": "đỏ và trắng"},
    "P_toc_do_toi_da":           {"desc": "biển tốc độ tối đa",                     "group": "biển cấm",       "action": "giảm tốc độ không vượt quá giới hạn",            "color": "đỏ và trắng"},
    "R_huong_phai_di":           {"desc": "biển hướng đi phải theo",                "group": "biển hiệu lệnh", "action": "đi theo hướng chỉ định trên biển",               "color": "xanh và trắng"},
    "R_khu_dong_dan_cu":         {"desc": "biển khu đông dân cư",                   "group": "biển hiệu lệnh", "action": "giảm tốc độ vì vào khu đông dân cư",             "color": "xanh và trắng"},
    "R_lan_duong_danh_cho_xe":   {"desc": "biển làn đường dành riêng",              "group": "biển hiệu lệnh", "action": "đi đúng làn đường theo quy định",                "color": "xanh và trắng"},
    "R_vong_chuong_ngai_vat":    {"desc": "biển vòng qua chướng ngại vật",          "group": "biển hiệu lệnh", "action": "đi vòng qua chướng ngại vật theo hướng chỉ định","color": "xanh và trắng"},
    "R_vong_xuyen":              {"desc": "biển vòng xuyến",                        "group": "biển hiệu lệnh", "action": "chạy theo chiều vòng xuyến",                     "color": "xanh và trắng"},
    "S_thuyet_minh":             {"desc": "biển phụ thuyết minh",                   "group": "biển phụ",       "action": "đọc thêm thông tin bổ sung",                     "color": "trắng và đen"},
    "W_cong_truong":             {"desc": "biển cảnh báo có công trường",            "group": "biển cảnh báo",  "action": "giảm tốc độ và chú ý công trường phía trước",    "color": "vàng và đen"},
    "W_di_cham":                 {"desc": "biển đi chậm",                           "group": "biển cảnh báo",  "action": "giảm tốc độ đi chậm lại",                       "color": "vàng và đen"},
    "W_doc_nguy_hiem":           {"desc": "biển dốc nguy hiểm",                     "group": "biển cảnh báo",  "action": "giảm tốc độ vì có dốc nguy hiểm phía trước",     "color": "vàng và đen"},
    "W_duong_doi":               {"desc": "biển đường đôi",                         "group": "biển cảnh báo",  "action": "chú ý có đường đôi phía trước",                  "color": "vàng và đen"},
    "W_duong_giao_nhau":         {"desc": "biển đường giao nhau",                   "group": "biển cảnh báo",  "action": "giảm tốc độ và chú ý đường giao nhau",           "color": "vàng và đen"},
    "W_duong_thu_hep":           {"desc": "biển đường bị thu hẹp",                  "group": "biển cảnh báo",  "action": "chú ý đường phía trước bị thu hẹp",              "color": "vàng và đen"},
    "W_duong_xau":               {"desc": "biển đường xấu hoặc có gồ giảm tốc",    "group": "biển cảnh báo",  "action": "giảm tốc độ vì đường xấu hoặc có gồ",            "color": "vàng và đen"},
    "W_giao_nhau_den_tin_hieu":  {"desc": "biển giao nhau có tín hiệu đèn",         "group": "biển cảnh báo",  "action": "chú ý đèn tín hiệu giao thông phía trước",       "color": "vàng và đen"},
    "W_giao_nhau_duong_nhanh":   {"desc": "biển giao nhau với đường không ưu tiên", "group": "biển cảnh báo",  "action": "chú ý đường giao nhau, xe chính được ưu tiên",   "color": "vàng và đen"},
    "W_giao_nhau_duong_sat":     {"desc": "biển giao nhau với đường sắt",           "group": "biển cảnh báo",  "action": "dừng lại và chú ý tàu hỏa",                      "color": "vàng và đen"},
    "W_giao_nhau_duong_uu_tien": {"desc": "biển giao nhau với đường ưu tiên",       "group": "biển cảnh báo",  "action": "nhường đường cho xe trên đường ưu tiên",         "color": "vàng và đen"},
    "W_ngoat_nguy_hiem":         {"desc": "biển chỗ ngoặt nguy hiểm",               "group": "biển cảnh báo",  "action": "giảm tốc độ vì có khúc cua nguy hiểm",           "color": "vàng và đen"},
    "W_nguy_hiem_khac":          {"desc": "biển nguy hiểm khác",                    "group": "biển cảnh báo",  "action": "giảm tốc độ và chú ý nguy hiểm phía trước",      "color": "vàng và đen"},
    "W_nhieu_ngoat":             {"desc": "biển nhiều chỗ ngoặt liên tiếp",         "group": "biển cảnh báo",  "action": "giảm tốc độ vì có nhiều khúc cua liên tiếp",     "color": "vàng và đen"},
    "W_nguoi_di_bo":             {"desc": "biển đường người đi bộ cắt ngang",       "group": "biển cảnh báo",  "action": "nhường đường cho người đi bộ",                   "color": "vàng và đen"},
    "W_tre_em":                  {"desc": "biển cảnh báo trẻ em",                   "group": "biển cảnh báo",  "action": "giảm tốc độ và chú ý trẻ em",                    "color": "vàng và đen"},
}


def get_pos_h(cx):
    if cx < 0.33: return "bên trái"
    elif cx < 0.66: return "ở giữa"
    else: return "bên phải"

def get_pos_v(cy):
    if cy < 0.4: return "phía trên"
    elif cy < 0.7: return "giữa ảnh"
    else: return "phía dưới"

def get_size(w, h):
    area = w * h
    if area > 0.04: return "lớn"
    elif area > 0.01: return "vừa"
    else: return "nhỏ"


def generate_templates(image_path, annotations, class_names):
    samples = []
    if not annotations:
        return samples

    signs = []
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) < 5:
            continue
        cls_idx = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        class_name = class_names[cls_idx]
        info = CLASS_INFO.get(class_name)
        if not info:
            continue
        signs.append({
            "class_name": class_name,
            "desc": info["desc"],
            "group": info["group"],
            "action": info["action"],
            "color": info["color"],
            "pos_h": get_pos_h(cx),
            "pos_v": get_pos_v(cy),
            "size": get_size(w, h),
            "cx": cx, "cy": cy,
        })

    if not signs:
        return samples

    img = str(image_path)
    n = len(signs)
    classes_present = {s["class_name"] for s in signs}
    groups = defaultdict(list)
    for s in signs:
        groups[s["group"]].append(s)

    # 1. ĐẾM
    samples.append({"image": img, "question": "Có bao nhiêu biển báo giao thông trong ảnh?", "answer": str(n), "type": "count"})
    for grp, gs in groups.items():
        samples.append({"image": img, "question": f"Có bao nhiêu {grp} trong ảnh?", "answer": str(len(gs)), "type": "count"})

    # 2. NHẬN DẠNG
    for s in signs:
        samples.append({"image": img, "question": f"Biển báo {s['pos_h']} trong ảnh là gì?", "answer": s["desc"].capitalize(), "type": "recognition"})
        samples.append({"image": img, "question": f"Biển báo {s['pos_h']} thuộc nhóm nào?", "answer": s["group"].capitalize(), "type": "recognition"})
        samples.append({"image": img, "question": f"Biển báo {s['pos_h']} có kích thước thế nào?", "answer": f"Biển {s['size']}", "type": "recognition"})

    # 3. YES/NO
    for s in signs:
        samples.append({"image": img, "question": f"Trong ảnh có {s['desc']} không?", "answer": "Có", "type": "yes_no"})
        samples.append({"image": img, "question": f"Đây có phải {s['group']} không?", "answer": "Có", "type": "yes_no"})

    absent = [c for c in CLASS_INFO.keys() if c not in classes_present]
    for c in random.sample(absent, min(3, len(absent))):
        samples.append({"image": img, "question": f"Trong ảnh có {CLASS_INFO[c]['desc']} không?", "answer": "Không", "type": "yes_no"})

    # 4. THUỘC TÍNH
    for s in signs:
        samples.append({"image": img, "question": f"Khi gặp {s['desc']}, tài xế cần làm gì?", "answer": s["action"].capitalize(), "type": "attribute"})
        samples.append({"image": img, "question": f"{s['desc'].capitalize()} có màu sắc chủ đạo là gì?", "answer": s["color"].capitalize(), "type": "attribute"})

    # 5. KHÔNG GIAN
    for s in signs:
        samples.append({"image": img, "question": f"{s['desc'].capitalize()} nằm ở vị trí nào trong ảnh?", "answer": f"{s['pos_h'].capitalize()}, {s['pos_v']}", "type": "spatial"})

    if n >= 2:
        leftmost = min(signs, key=lambda s: s["cx"])
        rightmost = max(signs, key=lambda s: s["cx"])
        topmost = min(signs, key=lambda s: s["cy"])
        samples.append({"image": img, "question": "Biển báo nào nằm ngoài cùng bên trái trong ảnh?", "answer": leftmost["desc"].capitalize(), "type": "spatial"})
        samples.append({"image": img, "question": "Biển báo nào nằm ngoài cùng bên phải trong ảnh?", "answer": rightmost["desc"].capitalize(), "type": "spatial"})
        samples.append({"image": img, "question": "Biển báo nào nằm cao nhất trong ảnh?", "answer": topmost["desc"].capitalize(), "type": "spatial"})

    return samples


def generate_template_dataset(data_dir, output_path):
    data_dir = Path(data_dir)
    with open(data_dir / "data.yaml") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    all_samples = []
    processed = 0

    for split in ["train", "valid", "test"]:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if not lbl_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            anns = [a for a in lbl_path.read_text().strip().split("\n") if a.strip()]
            all_samples.extend(generate_templates(img_path, anns, class_names))
            processed += 1

    random.seed(42)
    random.shuffle(all_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(all_samples), "samples": all_samples}, f, ensure_ascii=False, indent=2)

    print(f"\nTemplate: {len(all_samples)} bộ QA từ {processed} ảnh → {output_path}")
    counts = defaultdict(int)
    for s in all_samples:
        counts[s["type"]] += 1
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c}")
    return all_samples


# ============================================================
# PARAPHRASE + REASONING VỚI LLM
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"


def call_ollama(prompt, retries=3):
    for i in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1024},
                "think": False
            }, timeout=300)
            if r.status_code == 200:
                return r.json()["response"]
        except Exception as e:
            print(f"  Retry {i+1}: {e}")
            time.sleep(2)
    return None


def parse_json(text):
    try:
        s = text.find("[")
        e = text.rfind("]") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except:
        pass
    return []


def paraphrase_dataset(template_path, output_path, max_samples=None):
    with open(template_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    if max_samples:
        samples = samples[:max_samples]

    print(f"Paraphrase {len(samples)} mẫu bằng {MODEL}...")

    enriched = []
    by_image = defaultdict(list)
    for s in samples:
        by_image[s["image"]].append(s)

    total = len(samples)
    done = 0

    for img, img_samples in by_image.items():

        # Paraphrase từng câu hỏi
        for s in img_samples:
            enriched.append(s)
            prompt = f"""Bạn là chuyên gia tiếng Việt về giao thông đường bộ.
Câu hỏi gốc: "{s['question']}"
Câu trả lời đúng: "{s['answer']}"

Viết 3 cách hỏi KHÁC NHAU, tự nhiên như người Việt hỏi thật, câu trả lời vẫn là "{s['answer']}".
Chỉ trả lời JSON, không giải thích:
[{{"question": "...", "answer": "{s['answer']}"}}, {{"question": "...", "answer": "{s['answer']}"}}, {{"question": "...", "answer": "{s['answer']}"}}]"""

            resp = call_ollama(prompt)
            if resp:
                for q in parse_json(resp):
                    if q.get("question") and q.get("answer"):
                        enriched.append({"image": img, "question": q["question"], "answer": q["answer"], "type": s["type"] + "_llm"})

            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} xử lý ({len(enriched)} tổng)")

        # Sinh câu hỏi reasoning cho ảnh
        recognition = [s for s in img_samples if s["type"] == "recognition" and "là gì" in s["question"]]
        if len(recognition) >= 1:
            signs_list = "\n".join([f"- {s['answer']}" for s in recognition[:5]])
            prompt = f"""Bạn là chuyên gia luật giao thông Việt Nam.
Ảnh đường phố có các biển báo:
{signs_list}

Sinh 3 câu hỏi SUY LUẬN bằng tiếng Việt. Yêu cầu BẮT BUỘC:
- Câu hỏi phải liên quan trực tiếp đến các biển báo trong danh sách trên
- Câu trả lời chỉ dựa vào ý nghĩa biển báo, KHÔNG đề cập mức phạt tiền hay số liệu cụ thể
- Câu trả lời ngắn gọn dưới 10 từ, mô tả hành động hoặc quy định
- Câu hỏi phải có câu trả lời rõ ràng, KHÔNG hỏi những điều mơ hồ

Ví dụ câu hỏi tốt: "Xe máy có được đi vào đoạn đường này không?" → "Không, xe máy bị cấm"
Ví dụ câu hỏi xấu (KHÔNG làm): "Bị phạt bao nhiêu tiền?" hoặc "Biển này có tác dụng không?"

Chỉ trả lời JSON:
[{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}]"""

            resp = call_ollama(prompt)
            if resp:
                for q in parse_json(resp):
                    if q.get("question") and q.get("answer"):
                        enriched.append({"image": img, "question": q["question"], "answer": q["answer"], "type": "reasoning"})

    random.seed(42)
    random.shuffle(enriched)
    n = len(enriched)
    t1, t2 = int(n * 0.8), int(n * 0.9)

    output = {
        "metadata": {"total": n, "train": t1, "val": t2 - t1, "test": n - t2, "model": MODEL},
        "train": enriched[:t1],
        "val": enriched[t1:t2],
        "test": enriched[t2:]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== HOÀN THÀNH ===")
    print(f"Tổng: {n} | Train: {t1} | Val: {t2-t1} | Test: {n-t2}")
    counts = defaultdict(int)
    for s in enriched:
        counts[s["type"]] += 1
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["template", "paraphrase", "full"], required=True)
    parser.add_argument("--data_dir", help="Dataset remapped (cần cho template/full)")
    parser.add_argument("--input", help="File template JSON (cần cho paraphrase)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn mẫu để test nhanh")
    args = parser.parse_args()

    if args.mode == "template":
        assert args.data_dir, "Cần --data_dir"
        generate_template_dataset(args.data_dir, args.output)

    elif args.mode == "paraphrase":
        assert args.input, "Cần --input"
        paraphrase_dataset(args.input, args.output, args.max_samples)

    elif args.mode == "full":
        assert args.data_dir, "Cần --data_dir"
        tpl = args.output.replace(".json", "_template.json")
        print("=== BƯỚC 1: TEMPLATE ===")
        generate_template_dataset(args.data_dir, tpl)
        print("\n=== BƯỚC 2: LLM PARAPHRASE ===")
        paraphrase_dataset(tpl, args.output, args.max_samples)