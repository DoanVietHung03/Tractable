import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import evaluate

# File này đang ở: Modules/Visualize_metric/evaluate_metrics.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../Visualize_metric
parent_dir = os.path.dirname(current_dir)                # .../Modules
root_dir = os.path.dirname(parent_dir)                   # .../Tractable (Root)

# 2. Thêm root vào sys.path để Python nhìn thấy toàn bộ dự án
sys.path.append(root_dir)

import config

# 3. Import Class từ file train_segment.py (Dùng đường dẫn tuyệt đối từ Root)
try:
    from Modules.Segments.train_segment import SemanticSegmentationDataset, id2label, NUM_CLASSES
except ImportError:
    # Fallback: Nếu vẫn lỗi, thử import theo cách khác (phòng trường hợp cấu trúc folder khác)
    print("⚠️ Không import được từ Modules.Segments. Đang thử cách khác...")
    sys.path.append(os.path.join(root_dir, "Modules", "Segments"))
    from train_segment import SemanticSegmentationDataset, id2label, NUM_CLASSES

# --- CẤU HÌNH ---
MODEL_PATH = os.path.join(config.PROJECT_ROOT, "segformer_house_final")
TEST_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset", "test")

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa có model final!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Đang load model lên {device}...")
    
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
    processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    metric = evaluate.load("mean_iou")
    
    # Load tập Test
    test_dataset = SemanticSegmentationDataset(TEST_DIR, processor, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False) # Tăng batch nếu GPU mạnh
    
    print("🚀 Đang chạy đánh giá trên tập Test...")
    model.eval()
    
    for batch in tqdm(test_loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            
        # Post-process
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predictions = upsampled_logits.argmax(dim=1)
        
        # Đẩy vào metric để tính toán tích lũy
        metric.add_batch(
            predictions=predictions.detach().cpu().numpy(), 
            references=labels.detach().cpu().numpy()
        )
        
    # Tính kết quả cuối cùng
    results = metric.compute(num_labels=NUM_CLASSES, ignore_index=255, reduce_labels=False)
    
    print("\n📊 KẾT QUẢ FINAL:")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Accuracy: {results['overall_accuracy']:.4f}")
    
    # --- VẼ BIỂU ĐỒ IOU TỪNG CLASS ---
    ious = results["per_category_iou"]
    # Lọc bỏ các giá trị NaN (nếu class không xuất hiện trong tập test)
    valid_ious = [x if not np.isnan(x) else 0.0 for x in ious]
    
    class_names = [id2label[i] for i in range(len(valid_ious))]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, valid_ious, color='skyblue')
    
    # Tô màu đỏ cho cột nào dưới 0.5 (Yếu)
    for bar, val in zip(bars, valid_ious):
        if val < 0.5:
            bar.set_color('salmon')
        else:
            bar.set_color('mediumseagreen')
            
        # Hiện số lên đầu cột
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.2f}', ha='center', va='bottom')

    plt.title(f"IoU từng Class (Mean IoU: {results['mean_iou']:.2f})")
    plt.ylabel("IoU Score")
    plt.ylim(0, 1.05)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5) # Đường kẻ mốc 0.5
    plt.show()

if __name__ == "__main__":
    evaluate_model()