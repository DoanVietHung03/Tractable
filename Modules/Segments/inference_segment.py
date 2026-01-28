import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- 1. SETUP ĐỂ IMPORT CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

# --- 2. CẤU HÌNH ---
MODEL_PATH = os.path.join(config.PROJECT_ROOT, "segformer_house_final")
TEST_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset", "test", "images")
SAVE_DIR = os.path.join(config.OUTPUT_DIR) # Nơi lưu ảnh khi bấm 's'

NUM_SAMPLES = 10 # Load nhiều ảnh hơn để bấm space cho thoải mái

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

palette = [
    [0, 0, 0],       # 0: background
    [128, 0, 0],     # 1: building
    [0, 0, 128],     # 2: window
    [128, 128, 0],   # 3: door
    [0, 128, 0],     # 4: tree
    [0, 128, 128],   # 5: sky
    [128, 128, 128], # 6: road
    [128, 0, 128]    # 7: car
]

# --- 3. CLASS XỬ LÝ TƯƠNG TÁC ---
class InteractiveViewer:
    def __init__(self, model, processor, image_paths):
        self.model = model
        self.processor = processor
        self.image_paths = image_paths
        self.index = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Tạo thư mục lưu nếu chưa có
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Setup giao diện
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.manager.set_window_title('SegFormer Interactive Viewer')
        
        # Kết nối phím bấm
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n🎮 HƯỚNG DẪN SỬ DỤNG:")
        print("   [Space] : Xem ảnh tiếp theo")
        print("   [ s ]   : Lưu ảnh hiện tại")
        print("   [ q ]   : Thoát chương trình")
        
        # Hiển thị ảnh đầu tiên
        self.update_plot()
        plt.show()

    def colorize_mask(self, mask):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id, color in enumerate(palette):
            color_mask[mask == label_id] = color
        return color_mask

    def update_plot(self):
        # 1. Dọn dẹp plot cũ
        self.axs[0].clear()
        self.axs[1].clear()
        
        if self.index >= len(self.image_paths):
            print("Đã xem hết danh sách ảnh!")
            plt.close()
            return

        # 2. Lấy đường dẫn ảnh hiện tại
        img_path = self.image_paths[self.index]
        img_name = os.path.basename(img_path)
        print(f"\rOf [{self.index+1}/{len(self.image_paths)}]: Đang xử lý {img_name}...", end="")

        # 3. Dự đoán
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        color_pred = self.colorize_mask(pred_seg)

        # 4. Vẽ lại
        self.axs[0].imshow(image)
        self.axs[0].set_title(f"[{self.index+1}/{len(self.image_paths)}] Gốc: {img_name}", fontsize=12)
        self.axs[0].axis('off')

        self.axs[1].imshow(color_pred)
        self.axs[1].set_title("Segmentation", fontsize=12)
        self.axs[1].axis('off')

        # Vẽ chú thích (Legend) - Chỉ cần tạo 1 lần hoặc vẽ lại
        patches = [mpatches.Patch(color=np.array(palette[i])/255, label=label) for i, label in id2label.items()]
        self.fig.legend(handles=patches, loc='lower center', ncol=8, fontsize=10, frameon=False)
        
        self.fig.canvas.draw()

    def save_current_image(self):
        img_name = os.path.basename(self.image_paths[self.index])
        save_path = os.path.join(SAVE_DIR, f"result_{img_name}")
        self.fig.savefig(save_path)
        print(f"\n✅ Đã lưu ảnh tại: {save_path}")
        self.axs[1].set_title(f"Segmentation (ĐÃ LƯU)", color='green', fontweight='bold')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == ' ' or event.key == 'right': # Phím Space hoặc Mũi tên phải
            self.index += 1
            self.update_plot()
        elif event.key == 's': # Phím s
            self.save_current_image()
        elif event.key == 'q' or event.key == 'escape': # Phím q hoặc ESC
            plt.close()

# --- 4. MAIN ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa có model.")
        exit()

    if not os.path.exists(TEST_DIR):
        print("❌ Chưa có ảnh test.")
        exit()

    try:
        print("⏳ Đang load model...")
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        
        all_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png'))]
        
        if all_images:
            # Lấy mẫu ngẫu nhiên hoặc lấy hết
            sample_count = min(len(all_images), NUM_SAMPLES)
            sample_images = np.random.choice(all_images, sample_count, replace=False)
            
            # Khởi chạy Viewer
            viewer = InteractiveViewer(model, processor, sample_images)
        else:
            print("⚠️ Không tìm thấy ảnh nào.")
            
    except Exception as e:
        print(f"Lỗi: {e}")