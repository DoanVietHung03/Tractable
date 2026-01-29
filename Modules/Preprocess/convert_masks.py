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
    Mapping dựa trên bảng màu RGB của IRFS (image_e4d576.png):
    - Facade (128, 0, 0) -> Building
    - Window (0, 128, 0) -> Window
    - Door (128, 128, 0) -> Door
    - Roof (0, 0, 128)   -> Building
    - Balcony (128, 0, 128) -> Building
    - Shop (0, 128, 128) -> Door (Vì shop thường là cửa kính lớn)
    - Chimney (128, 128, 128) -> Building
    """
    try:
        # QUAN TRỌNG: Phải convert sang RGB để so sánh màu
        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)
    except Exception as e:
        print(f"Lỗi đọc file {mask_path}: {e}")
        return None

    h, w, _ = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)

    # --- MAPPING MÀU RGB -> ID ---
    
    # 1. Facade -> Building (1)
    # Tìm các pixel có màu (128, 0, 0)
    new_mask[(mask_np == (128, 0, 0)).all(axis=2)] = 1
    
    # 2. Window -> Window (2)
    new_mask[(mask_np == (0, 128, 0)).all(axis=2)] = 2
    
    # 3. Door -> Door (3)
    new_mask[(mask_np == (128, 128, 0)).all(axis=2)] = 3
    
    # 4. Roof -> Building (1)
    new_mask[(mask_np == (0, 0, 128)).all(axis=2)] = 1
    
    # 5. Balcony -> Building (1)
    new_mask[(mask_np == (128, 0, 128)).all(axis=2)] = 1
    
    # 6. Shop -> Door (3)
    new_mask[(mask_np == (0, 128, 128)).all(axis=2)] = 3
    
    # 7. Chimney -> Building (1)
    new_mask[(mask_np == (128, 128, 128)).all(axis=2)] = 1

    # Các màu còn lại (bao gồm (0,0,0) Background) mặc định là 0
    
    return new_mask

# --- Các hàm khác giữ nguyên ---
def convert_etrims_mask(mask_path):
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except: return None
    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)
    new_mask[mask_np == 1] = 1 # Building
    new_mask[mask_np == 2] = 7 # Car
    new_mask[mask_np == 3] = 3 # Door
    new_mask[mask_np == 4] = 6 # Road
    new_mask[mask_np == 5] = 6 # Road
    new_mask[mask_np == 6] = 5 # Sky
    new_mask[mask_np == 7] = 4 # Tree
    new_mask[mask_np == 8] = 2 # Window
    return new_mask

def convert_cmp_mask(mask_path):
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    except: return None
    h, w = mask_np.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)
    new_mask[mask_np == 1] = 1 
    new_mask[mask_np == 2] = 1
    new_mask[mask_np == 3] = 1
    new_mask[mask_np == 4] = 1
    new_mask[mask_np == 5] = 2 
    new_mask[mask_np == 6] = 3 
    new_mask[mask_np == 7] = 2 
    new_mask[mask_np == 8] = 2 
    new_mask[mask_np == 9] = 1 
    new_mask[mask_np == 10] = 3 
    new_mask[mask_np == 12] = 5 
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

if __name__ == "__main__":
    # 1. IRFS
    process_dataset(
        "IRFS", 
        os.path.join(config.IRFS_DIR, "0-1-Label"),
        os.path.join(config.IRFS_DIR, "0-1-masks"), 
        convert_irfs_mask
    )
    
    print("\n🎉 ĐÃ CONVERT XONG TOÀN BỘ!")