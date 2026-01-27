from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load model pre-trained trên ADE20K (có 150 class)
# Model này khá nhẹ (b0), nếu cần chính xác hơn hãy đổi thành nvidia/segformer-b4-finetuned-ade-512-512
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Load ảnh của bạn
image = Image.open("./Samples/download.jpg")

# Inference
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# Upscale về kích thước gốc
logits = nn.functional.interpolate(logits,
                size=image.size[::-1], # (height, width)
                mode='bilinear',
                align_corners=False)

# Lấy class dự đoán
pred_seg = torch.argmax(logits, dim=1)[0]
pred_seg_np = pred_seg.detach().cpu().numpy()

# Map label (Ví dụ trong ADE20K: 0=wall, 1=building, 2=sky, ... cần check file config id2label)
print("Các class phát hiện được:", np.unique(pred_seg_np))

# Hiển thị kết quả
# Danh sách các màu rực rỡ, dễ phân biệt (RGB)
# Đỏ, Xanh lá, Xanh dương, Vàng, Cyan, Magenta, Cam, Tím đậm...
HIGH_CONTRAST_COLORS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
    (255, 165, 0), # Orange
    (128, 0, 128), # Purple
    (255, 192, 203),# Pink
    (0, 128, 128)  # Teal
]

# Lấy map từ ID sang tên Class (Label)
id2label = model.config.id2label

# Tìm các class duy nhất xuất hiện trong ảnh này
unique_classes = np.unique(pred_seg_np)

# Tạo dictionary map class ID sang màu cụ thể
class_to_color_map = {}
for i, class_id in enumerate(unique_classes):
    # Gán màu từ danh sách, nếu hết màu thì quay vòng lại
    color = HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]
    class_to_color_map[class_id] = color

# --- PHẦN 2: TẠO ẢNH MASK MÀU ---
h, w = pred_seg_np.shape
seg_color = np.zeros((h, w, 3), dtype=np.uint8)

# Tô màu cho từng vùng dựa trên map đã tạo
for class_id, color in class_to_color_map.items():
    seg_color[pred_seg_np == class_id] = color

# --- PHẦN 3: BLEND VÀ HIỂN THỊ ---
# Chuyển ảnh gốc từ PIL sang Numpy
image_np = np.array(image)

# Pha trộn: Tăng độ đậm của mask lên (alpha 0.5) để màu rực hơn
alpha = 0.5
vis_img = cv2.addWeighted(image_np, alpha, seg_color, 1 - alpha, 0)

# Hiển thị
plt.figure(figsize=(10, 5))
plt.imshow(vis_img)
plt.axis('off')

# Tạo chú thích (Legend) với đúng màu đã gán
patches = []
print("--- CHI TIẾT VÀ MÀU SẮC ---")
for class_id, color_rgb in class_to_color_map.items():
    label_name = id2label[class_id]
    # Matplotlib cần màu 0-1
    color_normalized = [c / 255.0 for c in color_rgb]
    patches.append(mpatches.Patch(color=color_normalized, label=f"{class_id}: {label_name}"))
    print(f"ID {class_id}: {label_name} -> Màu RGB: {color_rgb}")

plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
plt.tight_layout()
plt.show()