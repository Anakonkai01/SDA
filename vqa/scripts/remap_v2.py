"""
Script remap class names từ dataset v2 (mã QCVN) → taxonomy mới
Usage: python remap_v2.py --data_dir data/datasets/raw_v2 --output_dir data/datasets/remapped_v2
"""

import os
import shutil
import yaml
import argparse
from pathlib import Path

# ============================================================
# BẢNG MAPPING: mã QCVN → class name taxonomy mới
# ============================================================
CLASS_MAPPING = {
    # --- NHÓM DP ---
    "DP.135":   "DP_het_tat_ca_lenh_cam",

    # --- NHÓM P ---
    "P.102":    "P_cam_nguoc_chieu",
    "P.103a":   "P_cam_xe_oto",
    "P.103b":   "P_cam_xe_oto_re",
    "P.103c":   "P_cam_xe_oto_re",
    "P.104":    "P_cam_xe_may",
    "P.106a":   "P_cam_xe_tai",
    "P.106b":   "P_cam_xe_tai",
    "P.107a":   "P_cam_xe_khach",
    "P.112":    "P_cam_nguoi_di_bo",
    "P.115":    "P_han_che_trong_tai",
    "P.117":    "P_han_che_chieu_cao",
    "P.123a":   "P_cam_re_trai",
    "P.123b":   "P_cam_re_phai",
    "P.124a":   "P_cam_quay_dau",
    "P.124b":   "P_cam_quay_dau",
    "P.124c":   "P_cam_quay_dau",
    "P.125":    "P_cam_vuot",
    "P.127":    "P_toc_do_toi_da",
    "P.128":    "P_cam_coi",
    "P.130":    "P_cam_dung_do_xe",
    "P.131a":   "P_cam_dung_do_xe",
    "P.137":    "P_cam_re_ca_hai_chieu",
    "P.245a":   "P_cam_dung_do_xe",  # biển cấm dừng đỗ dạng khác

    # --- NHÓM R ---
    "R.301a":   "R_huong_phai_di",
    "R.301c":   "R_huong_phai_di",
    "R.301d":   "R_huong_phai_di",
    "R.301e":   "R_huong_phai_di",
    "R.302a":   "R_vong_chuong_ngai_vat",
    "R.302b":   "R_vong_chuong_ngai_vat",
    "R.303":    "R_vong_xuyen",
    "R.407a":   "I_duong_mot_chieu",
    "R.409":    "I_cho_quay_xe",
    "R.425":    "R_khu_dong_dan_cu",
    "R.434":    "R_lan_duong_danh_cho_xe",

    # --- NHÓM S ---
    "S.509a":   "S_thuyet_minh",

    # --- NHÓM W ---
    "W.201a":   "W_ngoat_nguy_hiem",
    "W.201b":   "W_ngoat_nguy_hiem",
    "W.202a":   "W_nhieu_ngoat",
    "W.202b":   "W_nhieu_ngoat",
    "W.203b":   "W_duong_thu_hep",
    "W.203c":   "W_duong_thu_hep",
    "W.205a":   "W_duong_giao_nhau",
    "W.205b":   "W_duong_giao_nhau",
    "W.205d":   "W_duong_giao_nhau",
    "W.207a":   "W_giao_nhau_duong_nhanh",
    "W.207b":   "W_giao_nhau_duong_nhanh",
    "W.207c":   "W_giao_nhau_duong_nhanh",
    "W.208":    "W_giao_nhau_duong_uu_tien",
    "W.209":    "W_giao_nhau_den_tin_hieu",
    "W.210":    "W_giao_nhau_duong_sat",
    "W.219":    "W_doc_nguy_hiem",
    "W.221b":   "W_duong_xau",
    "W.224":    "W_nguoi_di_bo",
    "W.225":    "W_tre_em",
    "W.227":    "W_cong_truong",
    "W.233":    "W_nguy_hiem_khac",
    "W.235":    "W_duong_doi",
    "W.245a":   "W_di_cham",
}

NEW_CLASSES = sorted(set(CLASS_MAPPING.values()))


def remap_label_file(label_path, old_names, new_class_to_idx):
    lines = Path(label_path).read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_idx = int(parts[0])
        old_name = old_names[old_idx]
        new_name = CLASS_MAPPING.get(old_name)
        if new_name is None:
            print(f"  CẢNH BÁO: '{old_name}' không có mapping → bỏ qua")
            continue
        new_idx = new_class_to_idx[new_name]
        new_lines.append(f"{new_idx} " + " ".join(parts[1:]))
    return "\n".join(new_lines)


def remap_dataset(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    with open(data_dir / "data.yaml") as f:
        data = yaml.safe_load(f)

    old_names = data["names"]
    new_class_to_idx = {name: idx for idx, name in enumerate(NEW_CLASSES)}

    print(f"Dataset gốc: {len(old_names)} classes")
    print(f"Dataset mới: {len(NEW_CLASSES)} classes")

    for split in ["train", "valid", "test"]:
        src_img = data_dir / split / "images"
        src_lbl = data_dir / split / "labels"
        dst_img = output_dir / split / "images"
        dst_lbl = output_dir / split / "labels"

        if not src_lbl.exists():
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img in src_img.glob("*"):
            shutil.copy2(img, dst_img / img.name)

        count = 0
        for lbl_file in src_lbl.glob("*.txt"):
            content = remap_label_file(lbl_file, old_names, new_class_to_idx)
            (dst_lbl / lbl_file.name).write_text(content)
            count += 1

        print(f"{split}: {count} files xử lý xong")

    new_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_yaml, f, allow_unicode=True, default_flow_style=False)

    print(f"\nHoàn thành! Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="data/datasets/remapped_v2")
    args = parser.parse_args()
    remap_dataset(args.data_dir, args.output_dir)
