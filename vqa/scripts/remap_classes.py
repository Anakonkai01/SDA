"""
Script remap class names từ dataset Roboflow cũ → taxonomy mới theo QCVN 41:2024
Usage: python remap_classes.py --data_dir path/to/dataset
"""

import os
import glob
import shutil
import yaml
import argparse
from pathlib import Path

# ============================================================
# BẢNG MAPPING: class cũ → class mới
# ============================================================
CLASS_MAPPING = {
    # --- NHÓM DP ---
    "DP_133_het_cam_vuot":                                              "DP_het_cam_vuot",

    # --- NHÓM I ---
    "I_407a_duong_mot_chieu":                                           "I_duong_mot_chieu",
    "I_408_noi_do_xe":                                                  "I_noi_do_xe",
    "I_409_cho_quay_xe":                                                "I_cho_quay_xe",
    "I_414":                                                            "I_chi_huong_duong",
    "I_417":                                                            "I_chi_huong_theo_xe",
    "I_418_loi_di_o_nhung_vi_tri_cam_re":                               "I_loi_di_vi_tri_cam_re",
    "I_423b_vi_tri_nguoi_di_bo_sang_ngang":                             "I_vi_tri_nguoi_di_bo",
    "I_423b_vi_tri_nguoi_di_bo_sang_ngang_ben_phai":                    "I_vi_tri_nguoi_di_bo",
    "I_434a_ben_xe_buyt":                                               "I_ben_xe_bus",
    "I_441a_bao_hieu_phia_truoc_co_cong_truong":                        "I_cong_trinh_dac_biet",

    # --- NHÓM P ---
    "P_102_cam_di_nguoc_chieu":                                         "P_cam_nguoc_chieu",
    "P_102_cam_nguoc_chieu":                                            "P_cam_nguoc_chieu",
    "P_103a_cam_xe_oto":                                                "P_cam_xe_oto",
    "P_103b_cam_xe_oto_re_phai":                                        "P_cam_xe_oto_re",
    "P_103c_cam_xe_oto_re_trai":                                        "P_cam_xe_oto_re",
    "P_104_cam_xe_may":                                                 "P_cam_xe_may",
    "P_106a_cam_xe_oto_tai":                                            "P_cam_xe_tai",
    "P_106b_cam_xe_oto_tai_theo_khoi_luong":                            "P_cam_xe_tai",
    "P_107_cam_xe_oto_khach_va_xe_oto_tai":                             "P_cam_xe_khach",
    "P_107a_cam_xe_oto_khach":                                          "P_cam_xe_khach",
    "P_112_cam_nguoi_di_bo":                                            "P_cam_nguoi_di_bo",
    "P_115_han_che_trong_tai_toan_bo_xe":                               "P_han_che_trong_tai",
    "P_117_gioi_han_chieu_cao":                                         "P_han_che_chieu_cao",
    "P_117_han_che_chieu_cao":                                          "P_han_che_chieu_cao",
    # LƯU Ý: P_123a là cấm rẽ TRÁI theo quy chuẩn
    # Dataset có 2 entry P_123a: một gán nhầm là re_phai
    # → cả 2 map về P_cam_re_trai, cần review thủ công sau
    "P_123a_cam_re_phai":                                               "P_cam_re_trai",  # CẢNH BÁO: có thể gán nhầm trong dataset gốc
    "P_123a_cam_re_trai":                                               "P_cam_re_trai",
    "P_123b_cam_re_phai":                                               "P_cam_re_phai",
    "P_124a_cam_quay_dau_xe":                                           "P_cam_quay_dau",
    "P_124b_cam_oto_quay_dau_xe":                                       "P_cam_quay_dau",
    "P_124c_cam_re_trai_va_quay_dau_xe":                                "P_cam_quay_dau",
    "P_127_toc_do_toi_da_cho_phep":                                     "P_toc_do_toi_da",
    "P_127c_bien_ghep_toc_do_toi_da_cho_phep_theo_phuong_tien_tren_tung_lan_duong": "P_toc_do_toi_da",
    "P_130_cam_dung_va_do":                                             "P_cam_dung_do_xe",
    "P_130_cam_dung_xe_va_do_xe":                                       "P_cam_dung_do_xe",
    "P_131a_cam_do_xe":                                                 "P_cam_dung_do_xe",
    "P_131c_cam_do_xe_ngay_chan":                                       "P_cam_dung_do_xe",
    "P_137_cam_re_trai_re_phai":                                        "P_cam_re_ca_hai_chieu",

    # --- NHÓM R ---
    "R.E_9a_cam_do_xe_trong_khu_vuc":                                   "R_khu_vuc_do_xe",
    "R_301a_huong_di_phai_theo_di_thang":                               "R_huong_phai_di",
    "R_301e_huong_di_phai_theo_re_trai_gap":                            "R_huong_phai_di",
    "R_302a_huong_phai_di_vong_chuong_ngai_vat_sang_phai":              "R_vong_chuong_ngai_vat",
    "R_303_noi_giao_nhau_chay_theo_vong_xuyen":                         "R_vong_xuyen",
    "R_403":                                                            "R_duong_danh_cho_xe",
    "R_411_huong_di_tren_moi_lan_duong_phai_theo":                      "R_huong_lan_duong",
    "R_412":                                                            "R_lan_duong_danh_cho_xe",
    "R_415a_bien_gop_lan_duong_theo_phuong_tien":                       "R_lan_duong_danh_cho_xe",

    # --- NHÓM S ---
    "S_501_pham_vi_tac_dung_cua_bien":                                  "S_pham_vi_tac_dung",
    "S_502_khoang_cach_den_doi_tuong_bao_hieu":                         "S_khoang_cach",
    "S_505a_loai_xe":                                                   "S_loai_xe",
    "S_508a_bieu_thi_thoi_gian":                                        "S_thoi_gian",
    "S_508b_bieu_thi_thoi_gian":                                        "S_thoi_gian",
    "S_509a_chieu_cao_an_toan":                                         "S_thuyet_minh",

    # --- NHÓM W ---
    "W_201a_cho_ngoat_nguy_hiem_vong_ben_trai":                         "W_ngoat_nguy_hiem",
    "W_201b_cho_ngoat_nguy_hiem_vong_ben_phai":                         "W_ngoat_nguy_hiem",
    "W_202a_nhieu_cho_ngoac_lien_tiep_vong_ben_trai":                   "W_nhieu_ngoat",
    "W_202b_nhieu_cho_ngoac_lien_tiep_vong_ben_phai":                   "W_nhieu_ngoat",
    "W_203c_duong_bi_thu_hep_ben_phai":                                 "W_duong_thu_hep",
    "W_204_duong_hai_chieu":                                            "W_duong_hai_chieu",
    "W_205c_duong_giao_nhau_ben_trai":                                  "W_duong_giao_nhau",
    "W_207a_giao_nhau_voi_duong_khong_uu_tien":                         "W_giao_nhau_duong_nhanh",
    "W_207b_giao_nhau_voi_duong_khong_uu_tien_ben_phai":                "W_giao_nhau_duong_nhanh",
    "W_207c_giao_nhau_voi_duong_khong_uu_tien_ben_trai":                "W_giao_nhau_duong_nhanh",
    "W_208_giao_nhau_voi_duong_uu_tien":                                "W_giao_nhau_duong_uu_tien",
    "W_209_giao_nhau_co_tin_hieu_den":                                  "W_giao_nhau_den_tin_hieu",
    "W_215a_ke_vuc_sau_phia_truoc":                                     "W_ke_vuc_sau",
    "W_221b_duong_co_go_giam_toc":                                      "W_duong_xau",
    "W_224_duong_nguoi_di_bo_cat_ngang":                                "W_nguoi_di_bo",
    "W_224_nguoi_di_bo_cat_ngang":                                      "W_nguoi_di_bo",
    "W_225_tre_em":                                                     "W_tre_em",
    "W_227_cong_truong":                                                "W_cong_truong",
    "W_239a_duong_cap_dien_o_phia_tren":                                "W_nguy_hiem_khac",
    "W_244_doan_duong_hay_xay_ra_tai_nan":                              "W_nguy_hiem_khac",
    "W_245a_di_cham":                                                   "W_di_cham",

    # --- OTHER (giữ nguyên để review thủ công) ---
    "other":                                                            "other",
}

# ============================================================
# Danh sách class mới (theo thứ tự cố định)
# ============================================================
NEW_CLASSES = sorted(set(CLASS_MAPPING.values()))


def remap_label_file(label_path, old_names, new_class_to_idx):
    """Remap một file .txt annotation YOLO"""
    lines = Path(label_path).read_text().strip().split("\n")
    new_lines = []
    skipped = 0

    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_idx = int(parts[0])
        old_name = old_names[old_idx]
        new_name = CLASS_MAPPING.get(old_name)

        if new_name is None:
            print(f"  CẢNH BÁO: class '{old_name}' không có trong mapping → bỏ qua")
            skipped += 1
            continue

        new_idx = new_class_to_idx[new_name]
        new_lines.append(f"{new_idx} " + " ".join(parts[1:]))

    return "\n".join(new_lines), skipped


def remap_dataset(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Đọc data.yaml
    yaml_path = data_dir / "data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    old_names = data["names"]
    print(f"Dataset gốc: {len(old_names)} classes")

    # Tạo new class index mapping
    new_class_to_idx = {name: idx for idx, name in enumerate(NEW_CLASSES)}
    print(f"Dataset mới: {len(NEW_CLASSES)} classes")

    # Thống kê mapping
    print("\n=== MAPPING SUMMARY ===")
    mapped = {}
    for old in old_names:
        new = CLASS_MAPPING.get(old, "KHÔNG CÓ MAPPING")
        if new not in mapped:
            mapped[new] = []
        mapped[new].append(old)

    for new_name, old_list in sorted(mapped.items()):
        if len(old_list) > 1:
            print(f"  {new_name} ← {old_list}")

    # Tạo output directory
    splits = ["train", "valid", "test"]
    for split in splits:
        src_img = data_dir / split / "images"
        src_lbl = data_dir / split / "labels"
        dst_img = output_dir / split / "images"
        dst_lbl = output_dir / split / "labels"

        if not src_lbl.exists():
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # Copy ảnh
        for img in src_img.glob("*"):
            shutil.copy2(img, dst_img / img.name)

        # Remap labels
        total_files = 0
        total_skipped = 0
        for lbl_file in src_lbl.glob("*.txt"):
            new_content, skipped = remap_label_file(
                lbl_file, old_names, new_class_to_idx
            )
            (dst_lbl / lbl_file.name).write_text(new_content)
            total_files += 1
            total_skipped += skipped

        print(f"\n{split}: {total_files} files, {total_skipped} annotations bị bỏ qua")

    # Ghi data.yaml mới
    new_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_yaml, f, allow_unicode=True, default_flow_style=False)

    print(f"\n=== HOÀN THÀNH ===")
    print(f"Output: {output_dir}")
    print(f"Số classes mới: {len(NEW_CLASSES)}")
    print(f"\nCÁC CLASS CÓ DATA:")
    print("(chạy xong rồi kiểm tra bằng lệnh: python check_classes.py --data_dir <output_dir>)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Thư mục dataset gốc (có data.yaml)")
    parser.add_argument("--output_dir", default="dataset_remapped", help="Thư mục output")
    args = parser.parse_args()

    remap_dataset(args.data_dir, args.output_dir)
