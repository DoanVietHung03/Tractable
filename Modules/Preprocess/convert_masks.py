import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm

# --- 1. SETUP ĐỂ IMPORT CONFIG ---
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

def convert_irfs_mask(mask_path):
    """
    Mapping chuẩn xác dựa trên ảnh check_mask_id (IRFS.jpg):
    - ID 0: Building (Tường, bê tông) -> Map về 1
    - ID 1: Sky (Bầu trời) -> Map về 5
    - ID 2: Window (Kính) -> Map về 2
    - ID 4: Door (Lối vào) -> Map về 3
    - ID 5: Tree (Cây) -> Map về 4
    """
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except Exception as e:
        print(f"Lỗi đọc file {mask_path}: {e}")
        return None

    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)

    # --- MAPPING IRFS -> FINAL ---
    new_mask[mask_np == 0] = 1  # ID 0 là Tường -> Building
    new_mask[mask_np == 1] = 5  # ID 1 là Trời -> Sky
    new_mask[mask_np == 2] = 2  # ID 2 là Kính -> Window
    new_mask[mask_np == 4] = 3  # ID 4 là Cửa -> Door
    new_mask[mask_np == 5] = 4  # ID 5 là Cây -> Tree
    
    # Các ID lạ khác (nếu có) sẽ mặc định là 0 (Background)
    
    return new_mask

def convert_etrims_mask(mask_path):
    """ETRIMS Mapping (Giữ nguyên vì đã chuẩn)"""
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except: return None

    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)

    new_mask[mask_np == 1] = 1 # Building
    new_mask[mask_np == 2] = 7 # Car
    new_mask[mask_np == 3] = 3 # Door
    new_mask[mask_np == 4] = 6 # Pavement -> Road
    new_mask[mask_np == 5] = 6 # Road -> Road
    new_mask[mask_np == 6] = 5 # Sky
    new_mask[mask_np == 7] = 4 # Vegetation -> Tree
    new_mask[mask_np == 8] = 2 # Window
    
    return new_mask

def convert_cmp_mask(mask_path):
    """CMP Mapping (Giữ nguyên)"""
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except: return None
        
    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)
    
    new_mask[mask_np == 1] = 1 # Wall -> Building
    new_mask[mask_np == 2] = 1
    new_mask[mask_np == 3] = 1
    new_mask[mask_np == 4] = 1
    new_mask[mask_np == 5] = 2 # Window
    new_mask[mask_np == 6] = 3 # Door
    new_mask[mask_np == 7] = 2 
    new_mask[mask_np == 8] = 2 
    new_mask[mask_np == 9] = 1 
    new_mask[mask_np == 10] = 3 
    new_mask[mask_np == 12] = 5 # Sky
    
    return new_mask

def process_dataset(name, input_dir, output_dir, func):
    print(f"\n🚀 Đang xử lý: {name}...")
    if not os.path.exists(input_dir):
        print(f"❌ Không tìm thấy: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    count = 0
    for f in tqdm(files, desc=f"Converting {name}"):
        in_path = os.path.join(input_dir, f)
        
        # Convert
        new_mask = func(in_path)
        
        if new_mask is not None:
            out_name = os.path.splitext(f)[0] + ".png"
            out_path = os.path.join(output_dir, out_name)
            Image.fromarray(new_mask).save(out_path)
            count += 1
            
    print(f"✅ Xong {name}: {count} ảnh.")

# ================= MAIN =================
if __name__ == "__main__":
    
    # 1. Xử lý IRFS (QUAN TRỌNG NHẤT)
    process_dataset(
        "IRFS", 
        os.path.join(config.IRFS_DIR, "0-1-Label"),
        os.path.join(config.IRFS_DIR, "0-1-masks"), 
        convert_irfs_mask
    )
    
    print("\n🎉 ĐÃ CONVERT XONG TOÀN BỘ!")