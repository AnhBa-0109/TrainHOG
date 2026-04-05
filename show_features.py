"""
Chạy với đường dẫn ảnh: chỉ in đặc trưng HOG và Color Histogram (không dự đoán loài cá).

- Bấm Run: đặt đường dẫn trong biến DUONG_DAN_ANH_MAC_DINH ở cuối file.
- Terminal: python show_features.py "D:\\ảnh\\ca_ngu.jpg"  (ưu tiên hơn biến mặc định)
"""
import argparse
import os
import sys

import cv2
import numpy as np

from feature_extraction import extract_features_for_display

sys.stdout.reconfigure(encoding="utf-8")

HOG_PARAMS = (
    "orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm=L2-Hys, "
    "ảnh xám 128×128"
)
COLOR_PARAMS = "3 kênh BGR, mỗi kênh 32 bin trong [0, 256), ảnh 128×128"

# Thang ký tự từ sáng → tối (ASCII + một số ký tự block an toàn trên UTF-8 terminal)
_ASCII_RAMP = " .'`^\",:;Il!i><-+_-?][}{1)(|/tfjrxnuvczmwqpdbkhao*#MW&8%B@$"


def _normalize01(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _float_map_to_ascii_lines(matrix_01, width, height):
    """matrix_01: 2D 0..1 → chuỗi nhiều dòng ASCII."""
    m = np.clip(matrix_01, 0, 1)
    if m.size == 0:
        return ["(rỗng)"]
    small = cv2.resize(m, (width, height), interpolation=cv2.INTER_AREA)
    n = len(_ASCII_RAMP) - 1
    lines = []
    for row in small:
        lines.append("".join(_ASCII_RAMP[int(np.clip(v, 0.0, 1.0) * n)] for v in row))
    return lines


def print_ascii_heatmap(title, matrix, width=52, height=26):
    """Heatmap 2D bất kỳ (đã có ý nghĩa không gian hoặc ảnh)."""
    m = _normalize01(matrix)
    print(f"\n   [{title}]  ({matrix.shape[1]}×{matrix.shape[0]} → ASCII {width}×{height})")
    for line in _float_map_to_ascii_lines(m, width, height):
        print("   " + line)


def hog_vector_to_spatial_energy(hog_f):
    """
    Gom vector HOG skimage thành lưới không gian ~ (15×15): mỗi ô = năng lượng (L2) của block.
    Khớp cấu hình: 128/8=16 ô, block 2×2 → 15 vị trí mỗi chiều, 2×2×9=36 số/block.
    """
    n = hog_f.size
    cells = 16 - 2 + 1  # 15
    block_dim = 2 * 2 * 9  # 36
    if n != cells * cells * block_dim:
        return None
    blocks = hog_f.reshape(cells, cells, block_dim)
    energy = np.linalg.norm(blocks, axis=2)
    return energy


def print_histogram_ascii(b_hist, g_hist, r_hist, bar_width=40):
    """Mỗi kênh một hàng: chỉ số bin + thanh █ theo tỉ lệ max kênh."""
    labels = ("B", "G", "R")
    for label, h in zip(labels, (b_hist, g_hist, r_hist), strict=True):
        h = np.asarray(h).ravel()
        mx = float(np.max(h)) if h.size else 1.0
        if mx <= 0:
            mx = 1.0
        print(f"\n   Kênh {label} (max={mx:.1f})")
        for i, v in enumerate(h):
            frac = min(1.0, float(v) / mx)
            filled = int(round(frac * bar_width))
            bar = "█" * filled + "·" * (bar_width - filled)
            print(f"   {i:2d} │{bar} {v:8.1f}")


def print_gray_numeric_grid(gray_128, rows=8, cols=12):
    """
    Lưới số: gom vùng trên ảnh xám 128×128 → rows×cols ô, mỗi ô in độ sáng trung bình 0–255.
    """
    g = gray_128.astype(np.float32)
    h, w = g.shape
    cell_h = h // rows
    cell_w = w // cols
    print(f"\n   [Ảnh xám 128×128 → lưới {rows}×{cols} ô, mỗi ô = giá trị TB 0–255]")
    for r in range(rows):
        ys = slice(r * cell_h, (r + 1) * cell_h if r < rows - 1 else h)
        parts = []
        for c in range(cols):
            xs = slice(c * cell_w, (c + 1) * cell_w if c < cols - 1 else w)
            val = float(np.mean(g[ys, xs]))
            parts.append(f"{val:5.0f}")
        print("   " + " ".join(parts))


def _print_vec_sample(name, vec, head=12, tail=4):
    print(f"   • {name}: shape {vec.shape}, dtype {vec.dtype}")
    print(f"     min={vec.min():.6f}  max={vec.max():.6f}  mean={vec.mean():.6f}  std={vec.std():.6f}")
    h = min(head, len(vec))
    t = min(tail, max(0, len(vec) - h))
    parts = [np.array2string(vec[:h], precision=4, suppress_small=True, max_line_width=120)]
    if t > 0 and len(vec) > h:
        parts.append(" ... ")
        parts.append(np.array2string(vec[-t:], precision=4, suppress_small=True, max_line_width=120))
    print(f"     mẫu: {''.join(parts)}")


def print_feature_report(image_path, *, show_ascii=True):
    data = extract_features_for_display(image_path)
    if data is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return None

    hog_f = data["hog"]
    color_f = data["color_histogram"]
    hog_vis = data["hog_visualization"]
    gray = data["gray_128"]

    print("\n" + "=" * 52)
    print("  ĐẶC TRƯNG HOG (Histogram of Oriented Gradients)")
    print("=" * 52)
    print(f"   Tham số: {HOG_PARAMS}")
    _print_vec_sample("Vector HOG (1D)", hog_f, head=12, tail=4)

    if show_ascii:
        energy = hog_vector_to_spatial_energy(hog_f)
        if energy is not None:
            print_ascii_heatmap("Ma trận năng lượng HOG theo ô (15×15, mỗi ô = ‖block‖₂)", energy, width=45, height=22)
        vis = np.asarray(hog_vis, dtype=np.float64)
        print_ascii_heatmap("Ảnh minh họa HOG (skimage hog …, visualize=True)", vis, width=52, height=26)

    print("\n" + "=" * 52)
    print("  COLOR HISTOGRAM (BGR — 32 bin / kênh)")
    print("=" * 52)
    print(f"   Tham số: {COLOR_PARAMS}")
    _print_vec_sample("Toàn bộ 96 chiều [B|G|R]", color_f, head=12, tail=0)

    b_hist = color_f[0:32]
    g_hist = color_f[32:64]
    r_hist = color_f[64:96]

    if show_ascii:
        print("\n   --- Biểu đồ cột ASCII (mỗi bin một hàng) ---")
        print_histogram_ascii(b_hist, g_hist, r_hist, bar_width=36)
    else:
        print("\n   Tách theo kênh (số):")
        _print_vec_sample("Kênh B (32 bin)", b_hist, head=32, tail=0)
        _print_vec_sample("Kênh G (32 bin)", g_hist, head=32, tail=0)
        _print_vec_sample("Kênh R (32 bin)", r_hist, head=32, tail=0)

    if show_ascii:
        print("\n" + "=" * 52)
        print("  ẢNH ĐẦU VÀO (xám) — LƯỚI SỐ (ô lớn, không phải từng pixel)")
        print("=" * 52)
        print_gray_numeric_grid(gray, rows=8, cols=12)

    print("=" * 52 + "\n")

    return {"hog": hog_f, "color_histogram": color_f}


def main(default_image_path=None):
    parser = argparse.ArgumentParser(
        description="In đặc trưng HOG và Color Histogram từ một ảnh (không chạy SVM)."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        metavar="PATH",
        help='Đường dẫn file ảnh. Trên Windows nên đặt trong ngoặc kép, ví dụ: "D:\\ảnh\\ca.png"',
    )
    parser.add_argument(
        "-i",
        "--image",
        dest="image_flag",
        default=None,
        help="Cách khác để truyền đường dẫn ảnh (tránh lỗi với dấu \\ trên CMD).",
    )
    parser.add_argument(
        "--no-ascii",
        action="store_true",
        help="Chỉ in thống kê + vector mẫu (bỏ heatmap, biểu đồ cột, lưới số).",
    )
    args = parser.parse_args()

    path = args.image_flag or args.image
    if not path:
        path = default_image_path
    if not path:
        parser.error(
            'Cần đường dẫn ảnh: đặt biến DUONG_DAN_ANH_MAC_DINH ở cuối file, '
            'hoặc: python show_features.py \"D:\\\\thư_mục\\\\ảnh.jpg\"'
        )
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"❌ Không tìm thấy file: {path}")
        sys.exit(1)

    print(f"Ảnh: {path}")
    print_feature_report(path, show_ascii=not args.no_ascii)


if __name__ == "__main__":
    # Đường dẫn ảnh khi chạy bằng nút Run (không truyền tham số dòng lệnh).
    # Đổi thành file ảnh cá ngừ của bạn. Để trống rỗng "" thì bắt buộc phải gõ lệnh trong terminal.
    DUONG_DAN_ANH_MAC_DINH = r"D:\Learn\Python\TrainHOG\images\mt1.png"

    main(default_image_path=DUONG_DAN_ANH_MAC_DINH or None)
