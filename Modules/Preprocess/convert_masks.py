import numpy as np
import os
import sys
from PIL import Image

# --- 1. SETUP ĐỂ IMPORT CONFIG ---
# Lấy đường dẫn file hiện tại, đi lùi ra 2 cấp (Modules/Preprocess -> Root)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

# --- ĐÍCH ĐẾN: 8 CLASS CHUẨN CỦA PROJECT ---
# 0: background
# 1: building
# 2: window
# 3: door
# 4: tree
# 5: sky
# 6: road
# 7: car

def convert_etrims_mask(mask_path):
    """
    ETRIMS: Ảnh index 8 bit.
    Mapping dựa trên kết quả debug:
    1->Build, 2->Car, 3->Door, 4->Road, 5->Road, 6->Sky, 7->Tree, 8->Window
    """
    try:
        # Load ảnh chế độ 'L' (Grayscale/Index) để lấy đúng giá trị ID pixel
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except Exception as e:
        print(f"⚠️ Lỗi đọc file {os.path.basename(mask_path)}: {e}")
        return None

    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8) # Mặc định là 0 (Background)

    # --- MAPPING ETRIMS -> FINAL ---
    new_mask[mask_np == 1] = 1 # Building -> Building
    new_mask[mask_np == 2] = 7 # Car -> Car
    new_mask[mask_np == 3] = 3 # Door -> Door
    new_mask[mask_np == 4] = 6 # Pavement -> Road
    new_mask[mask_np == 5] = 6 # Road -> Road
    new_mask[mask_np == 6] = 5 # Sky -> Sky
    new_mask[mask_np == 7] = 4 # Vegetation -> Tree
    new_mask[mask_np == 8] = 2 # Window -> Window
    
    return new_mask

def convert_irfs_mask(mask_path):
    """
    IRFS: Ảnh index.
    Lưu ý quan trọng: ID 0 của IRFS là Sky (Trời), khác với chuẩn chung!
    """
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except:
        return None

    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)

    # --- MAPPING IRFS -> FINAL ---
    new_mask[mask_np == 0] = 5 # Sky -> Sky (Đã sửa lỗi quan trọng này)
    new_mask[mask_np == 1] = 1 # Building -> Building
    new_mask[mask_np == 2] = 2 # Window -> Window
    new_mask[mask_np == 3] = 1 # Các chi tiết phụ -> Building
    new_mask[mask_np == 4] = 3 # Door -> Door
    new_mask[mask_np == 5] = 4 # Tree -> Tree

    return new_mask

def convert_cmp_mask(mask_path):
    """
    CMP: Thường là ảnh index hoặc RGB chuẩn.
    Ta dùng mapping chuẩn của CMP Facade Database.
    """
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except:
        return None
        
    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)
    
    # --- MAPPING CMP -> FINAL ---
    # 1-4: Các loại tường/cột -> Building
    new_mask[mask_np == 1] = 1 
    new_mask[mask_np == 2] = 1
    new_mask[mask_np == 3] = 1
    new_mask[mask_np == 4] = 1
    
    # 5, 7, 8: Các loại cửa sổ/rèm -> Window
    new_mask[mask_np == 5] = 2
    new_mask[mask_np == 7] = 2
    new_mask[mask_np == 8] = 2
    
    # 6, 10: Cửa đi, Cửa hàng -> Door
    new_mask[mask_np == 6] = 3
    new_mask[mask_np == 10] = 3
    
    # 9, 11: Ban công, trang trí -> Building
    new_mask[mask_np == 9] = 1
    new_mask[mask_np == 11] = 1
    
    # 12: Sky -> Sky
    new_mask[mask_np == 12] = 5
    
    return new_mask

def process_dataset(dataset_name, input_folder, output_folder, convert_func):
    print(f"\n🚀 Đang xử lý bộ: {dataset_name}...")
    print(f"   Input:  {input_folder}")
    print(f"   Output: {output_folder}")

    if not os.path.exists(input_folder):
        print(f"❌ LỖI: Không tìm thấy thư mục input: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    count = 0
    for f in files:
        in_path = os.path.join(input_folder, f)
        
        # Thực hiện convert
        new_mask = convert_func(in_path)
        
        if new_mask is not None:
            # Lưu file kết quả dưới dạng PNG (quan trọng để giữ đúng giá trị pixel)
            # Giữ nguyên tên file gốc, chỉ đảm bảo đuôi là .png
            out_name = os.path.splitext(f)[0] + ".png"
            out_path = os.path.join(output_folder, out_name)
            
            Image.fromarray(new_mask).save(out_path)
            count += 1
            
    print(f"✅ Đã convert thành công {count} mask của {dataset_name}.")

# ================= MAIN =================
if __name__ == "__main__":
    # 1. ETRIMS
    process_dataset(
        "ETRIMS",
        os.path.join(config.ETRIMS_DIR, "annotations"),
        os.path.join(config.ETRIMS_DIR, "masks"),
        convert_etrims_mask
    )

    # 2. IRFS
    # Input lấy từ folder Label (chứa ảnh mask gốc)
    process_dataset(
        "IRFS",
        os.path.join(config.IRFS_DIR, "0-1-Label"), 
        os.path.join(config.IRFS_DIR, "0-1-masks"),
        convert_irfs_mask
    )
    
    print("\n🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH CONVERT!")