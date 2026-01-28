import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- 1. CẤU HÌNH ---
# Đường dẫn đến Model đã train xong (Nằm ở root/segformer_house_final)
MODEL_PATH = os.path.join(config.PROJECT_ROOT, "segformer_house_final")

# Đường dẫn đến ảnh Test (Nằm ở root/Final_Dataset/test/images)
TEST_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset", "test", "images")

NUM_SAMPLES = 10 # Số lượng ảnh muốn test thử

# Map ID sang Tên (Phải khớp với lúc train)
id2label = {
    0: "background",
    1: "building",
    2: "window",
    3: "door",
    4: "tree",
    5: "sky",
    6: "road",
    7: "car"
}

# Bảng màu hiển thị (R, G, B) - Tự chọn màu cho dễ nhìn
# 0: Đen, 1: Đỏ đun, 2: Xanh dương, 3: Cam, 4: Xanh lá, 5: Xanh trời, 6: Xám, 7: Tím
palette = [
    [0, 0, 0],       # 0: background
    [128, 0, 0],     # 1: building
    [0, 0, 128],     # 2: window
    [128, 64, 0],    # 3: door
    [0, 128, 0],     # 4: tree
    [0, 128, 128],   # 5: sky
    [128, 128, 128], # 6: road
    [128, 0, 128]    # 7: car
]

# --- 2. HÀM XỬ LÝ ---
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def colorize_mask(mask, palette):
    """Chuyển mask 2D (class ID) thành ảnh màu RGB 3D"""
    # mask: (H, W) -> id
    # output: (H, W, 3) -> rgb
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label_id, color in enumerate(palette):
        color_mask[mask == label_id] = color
        
    return color_mask

def show_predictions(model, processor, image_paths):
    device = get_device()
    model.to(device)
    model.eval()
    
    for i, img_path in enumerate(image_paths):
        # --- XỬ LÝ ẢNH ---
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        color_pred = colorize_mask(pred_seg, palette)
        
        # --- VẼ HÌNH (Tạo figure riêng cho mỗi ảnh) ---
        fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # Kích thước lớn, dễ nhìn
        
        # Ảnh gốc
        axs[0].imshow(image)
        axs[0].set_title(f"[{i+1}/{len(image_paths)}] Ảnh Gốc: {os.path.basename(img_path)}", fontsize=14)
        axs[0].axis('off')
        
        # Kết quả
        axs[1].imshow(color_pred)
        axs[1].set_title("Kết quả Segmentation", fontsize=14)
        axs[1].axis('off')
        
        # Chú thích
        patches = [mpatches.Patch(color=np.array(palette[k])/255, label=label) 
                   for k, label in id2label.items()]
        # Đặt chú thích bên phải cho gọn
        fig.legend(handles=patches, loc='center right', title="Chú giải Class")
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85) # Chừa chỗ cho cái Legend bên phải
        
        print(f"🖼️ Đang hiển thị ảnh {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        plt.show()

# --- 3. CHẠY THỰC TẾ ---
if __name__ == "__main__":
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Model Path:   {MODEL_PATH}")
    print(f"Test Img Dir: {TEST_DIR}")
    
    if not os.path.exists(MODEL_PATH):
        print("\n❌ LỖI: Không tìm thấy folder model!")
        print("👉 Bạn đã chạy xong 'train.py' chưa?")
        exit()

    if not os.path.exists(TEST_DIR):
        print("\n❌ LỖI: Không tìm thấy folder ảnh test!")
        print("👉 Bạn đã chạy 'split_data.py' để tạo dataset chưa?")
        exit()
        
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        print("\n✅ Đã load model thành công!")
        
        # Lấy ngẫu nhiên file ảnh
        all_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png'))]
        if not all_images:
            print("Không tìm thấy ảnh nào trong thư mục test!")
        else:
            sample_count = min(len(all_images), NUM_SAMPLES)
            sample_images = np.random.choice(all_images, sample_count, replace=False)
            
            print(f"📸 Đang dự đoán trên {sample_count} ảnh ngẫu nhiên...")
            show_predictions(model, processor, sample_images)
            print("Xong!")
            
    except Exception as e:
        print(f"Lỗi: {e}")