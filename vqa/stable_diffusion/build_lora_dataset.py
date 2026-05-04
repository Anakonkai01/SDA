import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS = ROOT / "data" / "processed" / "metadata" / "objects.jsonl"
DEFAULT_OUTPUT = ROOT / "stable_diffusion" / "dataset" / "metadata.jsonl"


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def unique_join(values):
    seen = []
    for value in values:
        value = clean_value(value)
        if value and value not in seen:
            seen.append(value)
    return ", ".join(seen)


POSITION_EN = {
    "Bên trái": "on the left side of the image",
    "Bên phải": "on the right side of the image",
    "Chính giữa": "in the center of the image",
    "Ở giữa": "in the center of the image",
    "Phía trên": "near the top of the image",
    "Phía dưới": "near the bottom of the image",
    "Trên bên trái": "in the upper left area of the image",
    "Trên bên phải": "in the upper right area of the image",
    "Dưới bên trái": "in the lower left area of the image",
    "Dưới bên phải": "in the lower right area of the image",
}

SHAPE_EN = {
    "Hình tròn": "circular",
    "Hình tam giác": "triangular",
    "Hình chữ nhật": "rectangular",
    "Hình vuông": "square",
    "Hình bát giác": "octagonal",
    "Hình thoi": "diamond-shaped",
}

COLOR_EN = {
    "Đỏ và trắng": "red and white",
    "Vàng và đen": "yellow and black",
    "Xanh và trắng": "blue and white",
    "Đỏ, xanh và trắng": "red, blue and white",
    "Trắng và đen": "white and black",
}

GROUP_EN = {
    "biển cấm": "prohibitory",
    "biển cảnh báo": "warning",
    "biển hiệu lệnh": "mandatory",
    "biển chỉ dẫn": "guide",
    "biển phụ": "supplementary",
    "biển báo": "general",
}


def translate(value, mapping):
    value = clean_value(value)
    return mapping.get(value, value)


def object_phrase(obj):
    label = clean_value(obj.get("class_en")) or "traffic sign"
    position = translate(obj.get("relative_position"), POSITION_EN)
    shape = translate(obj.get("shape"), SHAPE_EN)
    color = translate(obj.get("color_hint"), COLOR_EN)
    group = translate(obj.get("group"), GROUP_EN)

    descriptors = []
    if color:
        descriptors.append(color)
    if shape:
        descriptors.append(shape)
    if group:
        descriptors.append(group)

    phrase = " ".join(descriptors)
    if phrase:
        phrase += f" traffic sign for {label.lower()}"
    else:
        phrase = f"traffic sign for {label.lower()}"
    if position:
        phrase += f" {position}"
    return phrase


def build_caption(row, max_objects=4):
    objects = row.get("objects", [])[:max_objects]
    count = len(row.get("objects", []))
    phrases = [object_phrase(obj) for obj in objects]

    if not phrases:
        return "A realistic street photo in Vietnam with traffic signs and an urban road safety scene."

    sign_text = "; ".join(phrases)
    sign_count = "one" if count == 1 else str(count)
    sign_word = "traffic sign" if count == 1 else "traffic signs"
    return (
        f"A realistic street photo in Vietnam with {sign_count} visible {sign_word}. "
        f"The scene includes {sign_text}. "
        "Natural daylight, urban road environment, documentary photography style, high detail."
    )


def resolve_image_path(image_path):
    path = ROOT / "data" / "processed" / image_path
    if path.exists():
        return path
    return None


def load_rows(objects_path, split):
    rows = []
    with objects_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            image_path = row.get("image_path")
            if not image_path or not resolve_image_path(image_path):
                continue
            if not row.get("objects"):
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build a Stable Diffusion LoRA caption dataset from VQA object metadata.")
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.objects, args.split)
    random.Random(args.seed).shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {
                "image": row["image_path"],
                "caption": build_caption(row, max_objects=args.max_objects),
                "image_id": row.get("image_id"),
                "split": row.get("split"),
                "num_objects": len(row.get("objects", [])),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
