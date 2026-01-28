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

# Cấu hình đường dẫn IRFS
IMG_DIR = os.path.join(config.IRFS_DIR, "0-0-Image")
MASK_DIR = os.path.join(config.IRFS_DIR, "0-1-Label")

def inspect_classes():
    # Lấy đại 1 file ảnh để kiểm tra
    files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    if not files:
        print("Không tìm thấy ảnh!")
        return
    
    # Chọn ảnh thứ 2 (hoặc thay đổi index tùy ý)
    target_file = files[2] 
    img_path = os.path.join(IMG_DIR, target_file)
    
    # Tìm mask tương ứng (thường là .png)
    mask_name = os.path.splitext(target_file)[0] + ".png"
    mask_path = os.path.join(MASK_DIR, mask_name)
    
    if not os.path.exists(mask_path):
        print(f"Không thấy mask: {mask_path}")
        return

    # Load dữ liệu
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path)) # Mask này chứa các số 0, 1, 2...
    
    unique_ids = np.unique(mask)
    print(f"🔍 Đang soi ảnh: {target_file}")
    print(f"🔍 Các ID tìm thấy trong ảnh này: {unique_ids}")
    
    # Vẽ hình
    num_classes = len(unique_ids)
    rows = (num_classes + 1) // 3 + 1
    plt.figure(figsize=(15, 5 * rows))
    
    # Hình 1: Ảnh gốc
    plt.subplot(rows, 3, 1)
    plt.imshow(img)
    plt.title("Ảnh Gốc")
    plt.axis('off')
    
    # Các hình tiếp theo: Từng Class một
    for i, class_id in enumerate(unique_ids):
        # Tạo mask nhị phân: Chỗ nào bằng class_id thì sáng lên
        binary_mask = (mask == class_id).astype(np.uint8)
        
        plt.subplot(rows, 3, i + 2)
        plt.imshow(img) # Vẽ ảnh gốc làm nền
        plt.imshow(binary_mask, alpha=0.6, cmap='jet') # Vẽ mask đè lên (trong suốt)
        plt.title(f"Class ID: {class_id}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inspect_classes()