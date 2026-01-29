import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Setup đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
import config

# Đường dẫn đến dataset đã split
TRAIN_IMG_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset", "train", "images")
TRAIN_MASK_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset", "train", "masks")

# Bảng màu chuẩn của Project
palette = [
    [0, 0, 0],       # 0: Background (Đen)
    [128, 0, 0],     # 1: Building (Đỏ)
    [0, 0, 128],     # 2: Window (Xanh dương)
    [128, 128, 0],   # 3: Door (Vàng đất)
    [0, 128, 0],     # 4: Tree (Xanh lá)
    [0, 128, 128],   # 5: Sky (Xanh trời)
    [128, 128, 128], # 6: Road (Xám)
    [128, 0, 128]    # 7: Car (Tím)
]
labels = ["Back", "Build", "Win", "Door", "Tree", "Sky", "Road", "Car"]

def colorize(mask):
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        img[mask == i] = color
    return img

def check_data():
    if not os.path.exists(TRAIN_IMG_DIR):
        print("❌ Chưa có Final_Dataset. Hãy chạy split_data.py trước!")
        return

    # Lấy ngẫu nhiên 3 ảnh IRFS trong tập train để soi
    files = [f for f in os.listdir(TRAIN_IMG_DIR) if "irfs" in f.lower()][:3]
    
    if not files:
        print("⚠️ Không tìm thấy ảnh IRFS nào trong tập Train.")
        files = os.listdir(TRAIN_IMG_DIR)[:3] # Lấy ảnh bất kỳ

    plt.figure(figsize=(15, 10))
    print("🔍 Đang kiểm tra dữ liệu thực tế model sẽ học...")
    
    for i, fname in enumerate(files):
        # Load ảnh
        img = Image.open(os.path.join(TRAIN_IMG_DIR, fname))
        
        # Load mask
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(TRAIN_MASK_DIR, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"❌ Lỗi: Không thấy mask cho {fname}")
            continue
            
        mask = np.array(Image.open(mask_path))
        unique_ids = np.unique(mask)
        print(f"  - Ảnh {fname}: Tìm thấy Class IDs {unique_ids}")

        # Vẽ
        plt.subplot(3, 2, i*2 + 1)
        plt.imshow(img)
        plt.title(f"Ảnh gốc: {fname}")
        plt.axis('off')
        
        plt.subplot(3, 2, i*2 + 2)
        plt.imshow(colorize(mask))
        plt.title(f"Mask trong Final_Dataset\n(IDs: {unique_ids})")
        plt.axis('off')

    # Legend
    patches = [plt.Rectangle((0,0),1,1, color=np.array(c)/255) for c in palette]
    plt.legend(patches, labels, loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_data()