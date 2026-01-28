import os
import shutil
import random
import sys

# --- 1. SETUP ĐỂ IMPORT CONFIG ---
# Lấy đường dẫn file hiện tại, đi lùi ra 2 cấp (Modules/Preprocess -> Root) để tìm config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

# ================= CẤU HÌNH (Dùng biến từ config.py) =================

DATASET_CONFIGS = [
    {
        "name": "etrims",
        "img_dir": os.path.join(config.ETRIMS_DIR, "images"),
        "mask_dir": os.path.join(config.ETRIMS_DIR, "masks")
    },
    {
        "name": "irfs",
        "img_dir": os.path.join(config.IRFS_DIR, "0-0-Image"),
        "mask_dir": os.path.join(config.IRFS_DIR, "0-1-masks")
    }
]

# Output folder: Tạo folder "Final_Dataset" ngay tại thư mục gốc dự án
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ================= CÁC HÀM XỬ LÝ (GIỮ NGUYÊN) =================

def get_stem(filename):
    """Lấy tên file không bao gồm đuôi"""
    return os.path.splitext(filename)[0]

def find_mask_for_image(img_name, mask_files_map):
    """Tìm mask tương ứng cho ảnh (khớp tên bất kể đuôi)"""
    img_stem = get_stem(img_name)
    candidates = [
        img_stem + ".png",
        img_stem + ".jpg",
        img_stem + ".jpeg",
        img_stem + ".bmp",
        img_name 
    ]
    for cand in candidates:
        if cand in mask_files_map:
            return cand
    return None

def collect_files(configs):
    all_pairs = []
    print("--- BẮT ĐẦU QUÉT DỮ LIỆU ---")
    
    for conf in configs:
        name = conf["name"]
        img_dir = conf["img_dir"]
        mask_dir = conf["mask_dir"]
        
        # Kiểm tra tồn tại
        if not os.path.exists(img_dir):
            print(f"[CẢNH BÁO] Không tìm thấy thư mục ảnh: {img_dir}")
            continue
        if not os.path.exists(mask_dir):
            print(f"[CẢNH BÁO] Không tìm thấy thư mục mask: {mask_dir}")
            continue

        images = sorted(os.listdir(img_dir))
        masks = os.listdir(mask_dir)
        mask_map = {m: m for m in masks}
        
        count = 0
        missing_count = 0
        
        for img_name in images:
            if img_name.startswith('.'): continue # Bỏ qua file ẩn
            
            found_mask_name = find_mask_for_image(img_name, mask_map)
            
            if found_mask_name:
                pair = (
                    os.path.join(img_dir, img_name),
                    os.path.join(mask_dir, found_mask_name),
                    f"{name}_{img_name}" # Đổi tên để tránh trùng lặp giữa các bộ
                )
                all_pairs.append(pair)
                count += 1
            else:
                missing_count += 1
                if missing_count <= 3:
                    print(f"[MISSING] {name}: Không thấy mask cho '{img_name}'")

        print(f"Dataset '{name}': Đã ghép {count} cặp. (Thiếu mask: {missing_count})")
        
    return all_pairs

def copy_files(pairs, split_name):
    dest_img_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
    dest_mask_dir = os.path.join(OUTPUT_DIR, split_name, 'masks')
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)
    
    for src_img, src_mask, new_img_name in pairs:
        # Giữ đúng đuôi file của mask gốc
        mask_ext = os.path.splitext(src_mask)[1]
        new_mask_name = os.path.splitext(new_img_name)[0] + mask_ext
        
        shutil.copy2(src_img, os.path.join(dest_img_dir, new_img_name))
        shutil.copy2(src_mask, os.path.join(dest_mask_dir, new_mask_name))

def main():
    # In ra để debug đường dẫn
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Output Dir:   {OUTPUT_DIR}")

    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Dọn dẹp thư mục cũ: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    all_data = collect_files(DATASET_CONFIGS)
    total = len(all_data)
    print(f"\n--- TỔNG CỘNG: {total} CẶP DỮ LIỆU ---")

    if total == 0:
        print("❌ Không tìm thấy dữ liệu nào. Hãy kiểm tra lại config.py!")
        return

    # Shuffle và chia tập
    random.shuffle(all_data)
    train_c = int(total * TRAIN_RATIO)
    val_c = int(total * VAL_RATIO)
    
    train_set = all_data[:train_c]
    val_set = all_data[train_c : train_c + val_c]
    test_set = all_data[train_c + val_c :]

    print(f"Đang tạo dữ liệu tại: {OUTPUT_DIR}")
    copy_files(train_set, 'train')
    copy_files(val_set, 'val')
    copy_files(test_set, 'test')

    print("\n✅ HOÀN TẤT! Dữ liệu đã sẵn sàng.")

if __name__ == "__main__":
    main()