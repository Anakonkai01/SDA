# Danh sách Class VQA Dataset — Biển báo giao thông VN
**Chuẩn theo QCVN 41:2024/BGTVT**
**Mức: Loại (type-level) — có thể expand lên mức chi tiết sau**

---

## Nguyên tắc phân class

- Mỗi class = một **chức năng** biển báo, không phải từng mã biển
- Các biển cùng chức năng nhưng khác variant (a, b, c...) → **gộp 1 class**
- Các biển khác chức năng → **class riêng**
- Khi expand lên mức chi tiết → tách class thành subclass

---

## NHÓM P — BIỂN CẤM (28 classes)

| STT | Class name | Mã biển gốc | Ý nghĩa |
|---|---|---|---|
| 1 | `P_duong_cam` | P.101 | Đường cấm |
| 2 | `P_cam_nguoc_chieu` | P.102 | Cấm đi ngược chiều |
| 3 | `P_cam_xe_oto` | P.103a | Cấm xe ô tô |
| 4 | `P_cam_xe_oto_re` | P.103b, P.103c | Cấm xe ô tô rẽ trái/phải |
| 5 | `P_cam_xe_may` | P.104 | Cấm xe máy |
| 6 | `P_cam_xe_oto_va_xe_may` | P.105 | Cấm xe ô tô và xe máy |
| 7 | `P_cam_xe_tai` | P.106a, P.106b | Cấm xe ô tô tải |
| 8 | `P_cam_hang_nguy_hiem` | P.106c | Cấm xe chở hàng nguy hiểm |
| 9 | `P_cam_xe_khach` | P.107, P.107a, P.107b | Cấm xe khách / taxi |
| 10 | `P_cam_xe_ro_mooc` | P.108, P.108a | Cấm xe kéo rơ-moóc |
| 11 | `P_cam_xe_tho_so` | P.110a,b, P.111a,b,c,d | Cấm xe đạp, xe gắn máy, xích lô |
| 12 | `P_cam_nguoi_di_bo` | P.112 | Cấm người đi bộ |
| 13 | `P_han_che_trong_tai` | P.115, P.116 | Hạn chế trọng tải / tải trọng trục |
| 14 | `P_han_che_chieu_cao` | P.117 | Hạn chế chiều cao |
| 15 | `P_han_che_chieu_ngang` | P.118 | Hạn chế chiều ngang xe |
| 16 | `P_han_che_chieu_dai` | P.119, P.120 | Hạn chế chiều dài xe |
| 17 | `P_cu_ly_toi_thieu` | P.121 | Cự ly tối thiểu giữa hai xe |
| 18 | `P_cam_re_trai` | P.123a | Cấm rẽ trái |
| 19 | `P_cam_re_phai` | P.123b | Cấm rẽ phải |
| 20 | `P_cam_quay_dau` | P.124a,b,c,d,e,f | Cấm quay đầu xe |
| 21 | `P_cam_vuot` | P.125, P.126 | Cấm vượt |
| 22 | `P_toc_do_toi_da` | P.127, P.127a,b,c | Tốc độ tối đa cho phép |
| 23 | `P_cam_coi` | P.128 | Cấm sử dụng còi |
| 24 | `P_kiem_tra` | P.129 | Kiểm tra |
| 25 | `P_cam_dung_do_xe` | P.130, P.131a,b,c | Cấm dừng xe và đỗ xe |
| 26 | `P_nhuong_duong` | P.132 | Nhường đường |
| 27 | `P_cam_di_thang` | P.136, P.138, P.139 | Cấm đi thẳng (các variant) |
| 28 | `P_cam_re_ca_hai_chieu` | P.137 | Cấm rẽ trái và rẽ phải |

## NHÓM DP — HẾT CẤM (3 classes)

| STT | Class name | Mã biển gốc | Ý nghĩa |
|---|---|---|---|
| 29 | `DP_het_cam_vuot` | DP.133 | Hết cấm vượt |
| 30 | `DP_het_toc_do_toi_da` | DP.134, DP.127 | Hết tốc độ tối đa |
| 31 | `DP_het_tat_ca_lenh_cam` | DP.135 | Hết tất cả các lệnh cấm |

---

## NHÓM W — BIỂN CẢNH BÁO (23 classes)

| STT | Class name | Mã biển gốc | Ý nghĩa |
|---|---|---|---|
| 32 | `W_ngoat_nguy_hiem` | W.201a,b,c,d | Chỗ ngoặt nguy hiểm |
| 33 | `W_nhieu_ngoat` | W.202a,b | Nhiều chỗ ngoặt nguy hiểm liên tiếp |
| 34 | `W_duong_thu_hep` | W.203a,b,c | Đường bị thu hẹp |
| 35 | `W_duong_hai_chieu` | W.204, W.234 | Đường hai chiều |
| 36 | `W_duong_giao_nhau` | W.205a,b,c,d,e | Đường giao nhau |
| 37 | `W_giao_nhau_vong_xuyen` | W.206 | Giao nhau chạy theo vòng xuyến |
| 38 | `W_giao_nhau_duong_nhanh` | W.207 (tất cả variant) | Giao nhau với đường không ưu tiên |
| 39 | `W_giao_nhau_duong_uu_tien` | W.208 | Giao nhau với đường ưu tiên |
| 40 | `W_giao_nhau_den_tin_hieu` | W.209 | Giao nhau có tín hiệu đèn |
| 41 | `W_giao_nhau_duong_sat` | W.210, W.211, W.242, W.243 | Giao nhau với đường sắt |
| 42 | `W_cau_hep_tam` | W.212, W.213, W.214 | Cầu hẹp / cầu tạm / cầu quay |
| 43 | `W_ke_vuc_sau` | W.215a,b,c | Kè, vực sâu |
| 44 | `W_duong_ngam_ham` | W.216a,b, W.240 | Đường ngầm / đường hầm |
| 45 | `W_doc_nguy_hiem` | W.219, W.220 | Dốc xuống / dốc lên nguy hiểm |
| 46 | `W_duong_xau` | W.221a,b, W.222a,b | Đường lồi lõm, gồ giảm tốc, đường trơn |
| 47 | `W_nguoi_di_bo` | W.224 | Đường người đi bộ cắt ngang |
| 48 | `W_tre_em` | W.225 | Trẻ em |
| 49 | `W_xe_dap` | W.226 | Đường người đi xe đạp cắt ngang |
| 50 | `W_cong_truong` | W.227 | Công trường |
| 51 | `W_di_cham` | W.245a,b | Đi chậm |
| 52 | `W_un_tac` | W.241 | Ùn tắc giao thông |
| 53 | `W_duong_doi` | W.235, W.236, W.238 | Đường đôi / cao tốc phía trước |
| 54 | `W_nguy_hiem_khac` | W.223, W.228, W.229, W.230, W.231, W.232, W.233, W.244, W.246, W.247 | Nguy hiểm khác |

---

## NHÓM R — BIỂN HIỆU LỆNH (14 classes)

| STT | Class name | Mã biển gốc | Ý nghĩa |
|---|---|---|---|
| 55 | `R_dung_lai` | R.122 | Dừng lại |
| 56 | `R_huong_phai_di` | R.301a,b,c,d,e,f,g,h | Hướng đi phải theo |
| 57 | `R_vong_chuong_ngai_vat` | R.302a,b,c | Hướng phải đi vòng chướng ngại vật |
| 58 | `R_vong_xuyen` | R.303 | Nơi giao nhau chạy theo vòng xuyến |
| 59 | `R_duong_xe_tho_so` | R.304 | Đường dành cho xe thô sơ |
| 60 | `R_duong_nguoi_di_bo` | R.305 | Đường dành cho người đi bộ |
| 61 | `R_toc_do_toi_thieu` | R.306, R.307 | Tốc độ tối thiểu / hết tốc độ tối thiểu |
| 62 | `R_an_coi` | R.309 | Ấn còi |
| 63 | `R_duong_danh_cho_xe` | R.403a,b,c,d,e,f,g,h,k | Đường dành riêng cho loại xe |
| 64 | `R_het_duong_danh_cho_xe` | R.404a,b,c,d,e,f,g,h,k | Hết đoạn đường dành riêng |
| 65 | `R_huong_lan_duong` | R.411 | Hướng đi trên mỗi làn đường |
| 66 | `R_lan_duong_danh_cho_xe` | R.412a,b,c,d,e,f,g,h | Làn đường dành cho loại xe cụ thể |
| 67 | `R_khu_dong_dan_cu` | R.420, R.421 | Bắt đầu / hết khu đông dân cư |
| 68 | `R_khu_vuc_do_xe` | R.E9a,b,c,d, R.E10a,b,c,d | Cấm/cho phép đỗ xe trong khu vực |

---

## NHÓM I — BIỂN CHỈ DẪN (17 classes)

| STT | Class name | Mã biển gốc | Ý nghĩa |
|---|---|---|---|
| 69 | `I_duong_uu_tien` | I.401, I.402 | Bắt đầu / hết đường ưu tiên |
| 70 | `I_duong_cut` | I.405a,b,c | Đường cụt |
| 71 | `I_uu_tien_qua_duong_hep` | I.406 | Được ưu tiên qua đường hẹp |
| 72 | `I_duong_mot_chieu` | I.407a,b,c | Đường một chiều |
| 73 | `I_noi_do_xe` | I.408, I.408a | Nơi đỗ xe |
| 74 | `I_cho_quay_xe` | I.409, I.410 | Chỗ quay xe / khu vực quay xe |
| 75 | `I_chi_huong_duong` | I.414a,b,c,d, I.415 | Chỉ hướng đường / mũi tên chỉ hướng |
| 76 | `I_duong_tranh` | I.416 | Đường tránh |
| 77 | `I_chi_huong_theo_xe` | I.417a,b,c | Chỉ hướng đường theo loại xe |
| 78 | `I_loi_di_vi_tri_cam_re` | I.418 | Lối đi ở vị trí cấm rẽ |
| 79 | `I_chi_dan_dia_gioi` | I.419a,b | Chỉ dẫn địa giới |
| 80 | `I_vi_tri_nguoi_di_bo` | I.423a,b,c, I.424a,b | Vị trí người đi bộ sang ngang / cầu vượt |
| 81 | `I_dich_vu_y_te` | I.425, I.426 | Bệnh viện / trạm cấp cứu |
| 82 | `I_dich_vu_xe` | I.427a,b, I.428, I.429 | Sửa chữa / xăng dầu / rửa xe |
| 83 | `I_dich_vu_tien_ich` | I.430, I.431, I.432, I.433 | Điện thoại / trạm dừng / khách sạn |
| 84 | `I_ben_xe_bus` | I.434a,b, I.435 | Bến xe buýt / xe điện |
| 85 | `I_cong_trinh_dac_biet` | I.436, I.439, I.440, I.441, I.442, I.443, I.444, I.449 | Cảnh sát GT, tên cầu, đường thi công, chợ... |

---

## NHÓM S — BIỂN PHỤ (8 classes)

| STT | Class name | Mã biển gốc | Ý nghĩa |
|---|---|---|---|
| 86 | `S_pham_vi_tac_dung` | S.501 | Phạm vi tác dụng của biển |
| 87 | `S_khoang_cach` | S.502 | Khoảng cách đến đối tượng báo hiệu |
| 88 | `S_huong_tac_dung` | S.503a,b,c,d,e,f | Hướng tác dụng của biển |
| 89 | `S_lan_duong` | S.504 | Làn đường |
| 90 | `S_loai_xe` | S.505a,b,c | Loại xe |
| 91 | `S_huong_uu_tien` | S.506a,b, S.507 | Hướng đường ưu tiên / hướng rẽ |
| 92 | `S_thoi_gian` | S.508a,b | Biểu thị thời gian |
| 93 | `S_thuyet_minh` | S.509a,b, S.510b | Thuyết minh biển chính |

---

## Tổng kết

| Nhóm | Số class |
|---|---|
| P — Biển cấm | 28 |
| DP — Hết cấm | 3 |
| W — Biển cảnh báo | 23 |
| R — Biển hiệu lệnh | 14 |
| I — Biển chỉ dẫn | 17 |
| S — Biển phụ | 8 |
| **Tổng** | **93 classes** |

---

## Hướng mở rộng lên mức chi tiết

Khi có đủ data, mỗi class có thể tách thành subclass. Ví dụ:

```
P_toc_do_toi_da (1 class hiện tại)
    → P_toc_do_30
    → P_toc_do_40
    → P_toc_do_50
    → P_toc_do_60
    → P_toc_do_80
    → P_toc_do_100
    → P_toc_do_120

R_huong_phai_di (1 class hiện tại)
    → R_di_thang
    → R_re_phai
    → R_re_trai
    → R_di_thang_re_phai
    → R_di_thang_re_trai
```

---

*Chuẩn theo QCVN 41:2024/BGTVT — Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ*
