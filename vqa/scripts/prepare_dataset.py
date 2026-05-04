from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import ensure_dir, write_jsonl


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class ClassInfo:
    class_id: int
    class_code: str
    class_en: str
    class_vi: str
    group: str
    shape: str
    color_hint: str


@dataclass
class CandidateImage:
    source_image: Path
    source_label: Path
    original_name: str
    width: int
    height: int
    objects: list[dict[str, Any]]
    score: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_variant_marker(text: str) -> str:
    return text.replace("*", "").strip()


def infer_group(class_code: str, class_vi: str) -> str:
    prefix = class_code.split(".")[0].upper()
    if prefix == "P":
        return "biển cấm"
    if prefix == "W":
        return "biển cảnh báo"
    if prefix == "R":
        return "biển hiệu lệnh"
    if prefix == "I":
        return "biển chỉ dẫn"
    if prefix == "S":
        return "biển phụ"
    if prefix == "DP":
        return "biển hết cấm"
    text = class_vi.lower()
    if "cấm" in text:
        return "biển cấm"
    if "giao nhau" in text or "nguy hiểm" in text:
        return "biển cảnh báo"
    return "biển báo"


def infer_shape(class_code: str, class_vi: str) -> str:
    prefix = class_code.split(".")[0].upper()
    text = class_vi.lower()
    if prefix == "W" or "nguy hiểm" in text or "giao nhau" in text:
        return "Hình tam giác"
    if prefix in {"S", "I"} or "bến" in text or "làn" in text:
        return "Hình chữ nhật"
    return "Hình tròn"


def infer_color(class_code: str, class_vi: str) -> str:
    prefix = class_code.split(".")[0].upper()
    text = class_vi.lower()
    if prefix == "W" or "nguy hiểm" in text or "giao nhau" in text:
        return "Vàng và đen"
    if prefix in {"R", "I"}:
        return "Xanh và trắng"
    if prefix in {"S", "DP"}:
        return "Trắng và đen"
    return "Đỏ và trắng"


def load_class_infos(raw_root: Path) -> list[ClassInfo]:
    codes = read_lines(raw_root / "classes.txt")
    names_en = read_lines(raw_root / "classes_en.txt")
    names_vi = read_lines(raw_root / "classes_vie.txt")
    if not codes or len(codes) != len(names_vi):
        raise ValueError("Missing or inconsistent classes.txt/classes_vie.txt")

    infos: list[ClassInfo] = []
    for idx, code in enumerate(codes):
        class_code = strip_variant_marker(code)
        class_vi = names_vi[idx].strip()
        class_en = names_en[idx].strip() if idx < len(names_en) else class_code
        infos.append(ClassInfo(
            class_id=idx,
            class_code=class_code,
            class_en=class_en,
            class_vi=class_vi,
            group=infer_group(class_code, class_vi),
            shape=infer_shape(class_code, class_vi),
            color_hint=infer_color(class_code, class_vi),
        ))
    return infos


def relative_position(x: float, y: float, w: float, h: float, width: int, height: int) -> str:
    cx = x + w / 2
    cy = y + h / 2
    if cx < width / 3:
        x_zone = "trái"
    elif cx > 2 * width / 3:
        x_zone = "phải"
    else:
        x_zone = "giữa"

    if cy < height / 3:
        y_zone = "trên"
    elif cy > 2 * height / 3:
        y_zone = "dưới"
    else:
        y_zone = "giữa"

    mapping = {
        ("trên", "trái"): "Trên bên trái",
        ("trên", "phải"): "Trên bên phải",
        ("dưới", "trái"): "Dưới bên trái",
        ("dưới", "phải"): "Dưới bên phải",
        ("giữa", "trái"): "Bên trái",
        ("giữa", "phải"): "Bên phải",
        ("trên", "giữa"): "Phía trên",
        ("dưới", "giữa"): "Phía dưới",
        ("giữa", "giữa"): "Ở giữa",
    }
    return mapping[(y_zone, x_zone)]


def parse_label_file(
    label_path: Path,
    width: int,
    height: int,
    class_infos: list[ClassInfo],
    min_bbox_size: int,
    min_area_ratio: float,
) -> tuple[list[dict[str, Any]], float]:
    objects: list[dict[str, Any]] = []
    max_area_ratio = 0.0
    for line in read_lines(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(class_infos):
            continue
        cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
        bbox_w = max(1.0, w_norm * width)
        bbox_h = max(1.0, h_norm * height)
        x = max(0.0, (cx_norm - w_norm / 2) * width)
        y = max(0.0, (cy_norm - h_norm / 2) * height)
        bbox_w = min(bbox_w, width - x)
        bbox_h = min(bbox_h, height - y)
        area_ratio = (bbox_w * bbox_h) / (width * height)
        max_area_ratio = max(max_area_ratio, area_ratio)

        if bbox_w < min_bbox_size or bbox_h < min_bbox_size:
            continue
        if area_ratio < min_area_ratio:
            continue

        info = class_infos[class_id]
        objects.append({
            "class_id": class_id,
            "class_original": info.class_code,
            "class_en": info.class_en,
            "class_vi": info.class_vi,
            "group": info.group,
            "shape": info.shape,
            "color_hint": info.color_hint,
            "bbox_xywh": [round(x), round(y), round(bbox_w), round(bbox_h)],
            "bbox_yolo": [cx_norm, cy_norm, w_norm, h_norm],
            "relative_position": relative_position(x, y, bbox_w, bbox_h, width, height),
            "area_ratio": round(area_ratio, 6),
        })
    return objects, max_area_ratio


def load_split_names(split_path: Path) -> set[str]:
    names = set()
    for line in read_lines(split_path):
        names.add(Path(line).name)
    return names


def build_candidates(
    raw_root: Path,
    class_infos: list[ClassInfo],
    split_names: set[str] | None,
    min_bbox_size: int,
    min_area_ratio: float,
) -> list[CandidateImage]:
    image_dir = raw_root / "images"
    label_dir = raw_root / "labels"
    candidates: list[CandidateImage] = []
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if split_names is not None:
        image_paths = [p for p in image_paths if p.name in split_names]

    for image_path in tqdm(image_paths, desc=f"Scanning {image_dir.name}", leave=False):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            objects, max_area_ratio = parse_label_file(
                label_path,
                width,
                height,
                class_infos,
                min_bbox_size,
                min_area_ratio,
            )
        except Exception as exc:
            logging.warning("Skipping %s: %s", image_path, exc)
            continue
        if not objects:
            continue
        score = max_area_ratio + len(objects) * 0.001
        candidates.append(CandidateImage(
            source_image=image_path,
            source_label=label_path,
            original_name=image_path.name,
            width=width,
            height=height,
            objects=objects,
            score=score,
        ))
    candidates.sort(key=lambda c: (-c.score, c.original_name))
    return candidates


def pick_candidates(
    candidates: list[CandidateImage],
    count: int,
    seed: int,
    diversify_top_k: int,
) -> list[CandidateImage]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} images, only found {len(candidates)} eligible candidates")
    pool_size = max(count, min(len(candidates), diversify_top_k))
    pool = candidates[:pool_size]
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool[:count]
    selected.sort(key=lambda c: c.original_name)
    return selected


def copy_and_make_records(
    split: str,
    candidates: list[CandidateImage],
    out_dir: Path,
    start_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    object_records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    image_out_dir = ensure_dir(out_dir / "images" / split)

    next_index = start_index
    for candidate in tqdm(candidates, desc=f"Copying {split}"):
        image_id = f"vts_{next_index:06d}"
        next_index += 1
        dst_name = f"{image_id}{candidate.source_image.suffix.lower()}"
        dst_rel = Path("images") / split / dst_name
        dst_abs = out_dir / dst_rel
        shutil.copy2(candidate.source_image, dst_abs)

        objects = []
        for obj_idx, obj in enumerate(candidate.objects, start=1):
            copied = dict(obj)
            copied["object_id"] = f"{image_id}_o{obj_idx}"
            objects.append(copied)

        object_records.append({
            "image_id": image_id,
            "image_path": dst_rel.as_posix(),
            "width": candidate.width,
            "height": candidate.height,
            "split": split,
            "source": "kaggle_vnts",
            "original_image": candidate.original_name,
            "objects": objects,
        })
        image_rows.append({
            "image_id": image_id,
            "image_path": dst_rel.as_posix(),
            "split": split,
            "width": candidate.width,
            "height": candidate.height,
            "source": "kaggle_vnts",
            "original_image": candidate.original_name,
            "num_objects": len(objects),
        })
        split_rows.append({
            "image_id": image_id,
            "split": split,
            "original_image": candidate.original_name,
        })
    return object_records, image_rows, split_rows, next_index


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_class_map(path: Path, class_infos: list[ClassInfo]) -> None:
    rows = [
        {
            "class_id": info.class_id,
            "class_original": info.class_code,
            "class_en": info.class_en,
            "class_vi": info.class_vi,
            "group": info.group,
            "shape": info.shape,
            "color_hint": info.color_hint,
            "canonical_answer": info.class_vi,
        }
        for info in class_infos
    ]
    write_csv(path, rows, [
        "class_id",
        "class_original",
        "class_en",
        "class_vi",
        "group",
        "shape",
        "color_hint",
        "canonical_answer",
    ])


def write_stats(path: Path, object_records: list[dict[str, Any]]) -> None:
    images_by_split = Counter(r["split"] for r in object_records)
    objects_by_split = Counter()
    class_counts = Counter()
    object_counts = []
    for record in object_records:
        objects = record.get("objects") or []
        object_counts.append(len(objects))
        objects_by_split[record["split"]] += len(objects)
        for obj in objects:
            class_counts[obj.get("class_vi", "")] += 1

    stats = {
        "num_images": dict(images_by_split),
        "num_objects": dict(objects_by_split),
        "objects_per_image": {
            "min": min(object_counts) if object_counts else 0,
            "mean": sum(object_counts) / len(object_counts) if object_counts else 0,
            "max": max(object_counts) if object_counts else 0,
        },
        "class_distribution": dict(class_counts.most_common()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle VNTS data for VQA labeling.")
    parser.add_argument("--raw_dir", default="data/raw/kaggle_vnts/archive")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--num_images", type=int, default=350)
    parser.add_argument("--train_images", type=int, default=280)
    parser.add_argument("--val_images", type=int, default=35)
    parser.add_argument("--test_images", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_bbox_size", type=int, default=30)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--diversify_top_k", type=int, default=1200)
    parser.add_argument("--include_all", action="store_true", help="Use all eligible images instead of target counts")
    parser.add_argument(
        "--ignore_split_files",
        action="store_true",
        help="Scan all images in raw_dir/images. Images listed in test_files.txt stay in test; everything else goes to train.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    raw_root = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_root.exists():
        logging.error("raw_dir not found: %s", raw_root)
        return 1
    if args.train_images + args.val_images + args.test_images != args.num_images and not args.include_all:
        logging.error("train_images + val_images + test_images must equal num_images")
        return 1

    class_infos = load_class_infos(raw_root)
    train_names = load_split_names(raw_root / "split_dataset" / "train_files.txt")
    test_names = load_split_names(raw_root / "split_dataset" / "test_files.txt")
    logging.info("Loaded %s classes", len(class_infos))
    logging.info("Kaggle split names: train=%s test=%s", len(train_names), len(test_names))

    if args.ignore_split_files:
        all_candidates = build_candidates(
            raw_root, class_infos, None, args.min_bbox_size, args.min_area_ratio
        )
        test_candidates = [c for c in all_candidates if c.original_name in test_names]
        selected_train_pool = [c for c in all_candidates if c.original_name not in test_names]
        train_candidates = selected_train_pool
        logging.info(
            "Ignoring split file filtering: all_images=%s train_pool=%s test_pool=%s",
            len(all_candidates),
            len(train_candidates),
            len(test_candidates),
        )
    else:
        train_candidates = build_candidates(
            raw_root, class_infos, train_names, args.min_bbox_size, args.min_area_ratio
        )
        test_candidates = build_candidates(
            raw_root, class_infos, test_names, args.min_bbox_size, args.min_area_ratio
        )
    logging.info("Eligible candidates: train_pool=%s test_pool=%s", len(train_candidates), len(test_candidates))

    if args.include_all:
        selected_train = train_candidates
        selected_val: list[CandidateImage] = []
        selected_test = test_candidates
    else:
        train_val_needed = args.train_images + args.val_images
        selected_train_val = pick_candidates(
            train_candidates,
            train_val_needed,
            args.seed,
            args.diversify_top_k,
        )
        selected_train = selected_train_val[:args.train_images]
        selected_val = selected_train_val[args.train_images:]
        selected_test = pick_candidates(
            test_candidates,
            args.test_images,
            args.seed + 1,
            args.diversify_top_k,
        )

    if out_dir.exists():
        logging.warning("Output dir exists; files may be overwritten: %s", out_dir)
    ensure_dir(out_dir / "metadata")

    object_records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    next_index = 1
    for split, selected in [
        ("train", selected_train),
        ("val", selected_val),
        ("test", selected_test),
    ]:
        records, images, splits, next_index = copy_and_make_records(split, selected, out_dir, next_index)
        object_records.extend(records)
        image_rows.extend(images)
        split_rows.extend(splits)

    write_jsonl(out_dir / "metadata" / "objects.jsonl", object_records)
    write_csv(out_dir / "metadata" / "images.csv", image_rows, [
        "image_id",
        "image_path",
        "split",
        "width",
        "height",
        "source",
        "original_image",
        "num_objects",
    ])
    write_csv(out_dir / "metadata" / "split.csv", split_rows, ["image_id", "split", "original_image"])
    write_class_map(out_dir / "metadata" / "class_map.csv", class_infos)
    write_stats(out_dir / "metadata" / "stats.json", object_records)

    logging.info(
        "Prepared dataset: train=%s val=%s test=%s images",
        len(selected_train),
        len(selected_val),
        len(selected_test),
    )
    logging.info("Wrote metadata to %s", out_dir / "metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
