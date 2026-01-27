import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- 1. CẤU HÌNH ---
MODEL_PATH = "segformer_house_final"  # Folder chứa model đã train
TEST_DIR = "/kaggle/working/Final_Dataset/test/images" # Folder ảnh test
NUM_SAMPLES = 4 # Số lượng ảnh muốn test thử

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
    [0, 191, 255],   # 5: sky
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
    
    # Tạo plot
    fig, axs = plt.subplots(len(image_paths), 2, figsize=(15, 6 * len(image_paths)))
    if len(image_paths) == 1: axs = [axs] # Fix lỗi dimension nếu chỉ có 1 ảnh
    
    for i, img_path in enumerate(image_paths):
        # 1. Load ảnh
        image = Image.open(img_path).convert("RGB")
        
        # 2. Preprocess
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 3. Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 4. Post-process (Upsample logits về kích thước ảnh gốc)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width) - Lưu ý PIL size là (W, H)
            mode="bilinear",
            align_corners=False,
        )
        
        # Lấy class có xác suất cao nhất
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # 5. Tô màu
        color_pred = colorize_mask(pred_seg, palette)
        
        # 6. Hiển thị
        ax_curr = axs[i] if len(image_paths) > 1 else axs
        
        # Ảnh gốc
        ax_curr[0].imshow(image)
        ax_curr[0].set_title(f"Ảnh Gốc: {os.path.basename(img_path)}")
        ax_curr[0].axis('off')
        
        # Ảnh dự đoán
        ax_curr[1].imshow(color_pred)
        ax_curr[1].set_title("Kết quả Segmentation")
        ax_curr[1].axis('off')
        
    # Tạo chú thích (Legend)
    patches = [mpatches.Patch(color=np.array(palette[i])/255, label=label) 
               for i, label in id2label.items()]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12)
    plt.tight_layout()
    plt.show()

# --- 3. CHẠY THỰC TẾ ---
if __name__ == "__main__":
    print(f"Đang load model từ: {MODEL_PATH} ...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        
        # Lấy ngẫu nhiên file ảnh
        all_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png'))]
        if not all_images:
            print("Không tìm thấy ảnh nào trong thư mục test!")
        else:
            sample_images = np.random.choice(all_images, min(len(all_images), NUM_SAMPLES), replace=False)
            print("Đang dự đoán...")
            show_predictions(model, processor, sample_images)
            
    except Exception as e:
        print(f"Lỗi: {e}")
        print("Gợi ý: Hãy chắc chắn bạn đã chạy xong phần Train và folder 'segformer_house_final' đã tồn tại.")