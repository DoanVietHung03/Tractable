import cv2
import numpy as np
import os
import sys

# --- 1. SETUP ĐỂ IMPORT CONFIG ---
# Lấy đường dẫn file hiện tại, đi lùi ra 2 cấp (Modules/Preprocess -> Root)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

# ================= CẤU HÌNH MÀU SẮC (GIỮ NGUYÊN) =================
CLASS_COLOR_GROUPS = {
    0: [(0, 0, 0)], # Background
    1: [(128, 0, 0)], # Building 
    2: [(0, 0, 128)], # Window
    3: [(128, 128, 0), (128, 64, 0)], # Door
    4: [(0, 128, 0), (170, 255, 85)], # Tree
    5: [
        (0, 128, 128),
        (0, 0, 255),
        (0, 0, 170),
        (0, 85, 255),
        (0, 170, 255)
    ], # Sky
    6: [(128, 128, 128)], # Road
    7: [(128, 0, 128)] # Car
}

COLOR_MAP = {}
for class_id, colors in CLASS_COLOR_GROUPS.items():
    for c in colors:
        COLOR_MAP[c] = class_id


def rgb_mask_to_class_id(mask_rgb):
    h, w, _ = mask_rgb.shape
    class_mask = np.full((h, w), fill_value=255, dtype=np.uint8)  # 255 = invalid

    for rgb, class_id in COLOR_MAP.items():
        r, g, b = rgb
        match = (
            (mask_rgb[:, :, 0] == r) &
            (mask_rgb[:, :, 1] == g) &
            (mask_rgb[:, :, 2] == b)
        )
        class_mask[match] = class_id

    return class_mask


def find_unknown_colors(mask_rgb):
    known = set(COLOR_MAP.keys())
    colors = set(tuple(c) for c in mask_rgb.reshape(-1, 3))
    return colors - known

# ================= XỬ LÝ CHÍNH =================
if __name__ == "__main__":
    # SỬ DỤNG ĐƯỜNG DẪN TỪ CONFIG
    # Dựa vào cấu trúc thư mục của bạn: annotations nằm trong etrims
    input_dir  = os.path.join(config.ETRIMS_DIR, "annotations")
    output_dir = os.path.join(config.ETRIMS_DIR, "masks")
    
    print(f"Input Dir:  {input_dir}")
    print(f"Output Dir: {output_dir}")

    if not os.path.exists(input_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục input: {input_dir}")
        exit()

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg")):
            continue

        bgr = cv2.imread(os.path.join(input_dir, fname))
        if bgr is None:
            print(f"⚠️ Không đọc được ảnh: {fname}")
            continue
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        unknown = find_unknown_colors(rgb)
        if len(unknown) > 0:
            print(f"[ERROR] {fname} still has unknown colors: {unknown}")
            continue  # KHÔNG convert nếu còn màu lạ

        class_mask = rgb_mask_to_class_id(rgb)

        # Check an toàn lần cuối
        if 255 in np.unique(class_mask):
            print(f"Invalid pixel in {fname}")
            continue

        cv2.imwrite(os.path.join(output_dir, fname), class_mask)
        count += 1
        print(f"Converted OK: {fname}")

    print(f"\n✅ Đã convert xong {count} ảnh mask.")