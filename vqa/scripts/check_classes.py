"""
Script kiểm tra phân bố class sau khi remap
Usage: python check_classes.py --data_dir path/to/remapped_dataset
"""

import os
import yaml
import glob
from pathlib import Path
from collections import defaultdict
import argparse


def check_dataset(data_dir):
    data_dir = Path(data_dir)
    yaml_path = data_dir / "data.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    class_names = data["names"]
    class_counts = defaultdict(int)

    # Đếm annotations trong tất cả splits
    for split in ["train", "valid", "test"]:
        lbl_dir = data_dir / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in Path(lbl_file).read_text().strip().split("\n"):
                if line.strip():
                    idx = int(line.split()[0])
                    class_counts[class_names[idx]] += 1

    # In kết quả
    print(f"\n=== PHÂN BỐ CLASS ({data_dir.name}) ===")
    print(f"Tổng số classes: {len(class_names)}")
    print(f"Classes có data: {sum(1 for v in class_counts.values() if v > 0)}")
    print(f"Classes không có data: {sum(1 for v in class_counts.values() if v == 0)}\n")

    print(f"{'Class':<45} {'Số ảnh':>8} {'Đánh giá':>12}")
    print("-" * 70)

    for name in sorted(class_names):
        count = class_counts.get(name, 0)
        if count == 0:
            status = "⚠️  TRỐNG"
        elif count < 10:
            status = "⚠️  RẤT ÍT"
        elif count < 50:
            status = "⚡ ÍT"
        else:
            status = "✅ ĐỦ"
        print(f"{name:<45} {count:>8}    {status}")

    # Classes cần bổ sung
    need_more = [n for n in class_names if class_counts.get(n, 0) < 10]
    if need_more:
        print(f"\n=== CẦN BỔ SUNG ẢNH ({len(need_more)} classes) ===")
        for name in need_more:
            print(f"  - {name}: {class_counts.get(name, 0)} ảnh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    check_dataset(args.data_dir)
