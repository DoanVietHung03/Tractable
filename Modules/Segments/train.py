import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# --- 1. SETUP ĐỂ IMPORT CONFIG ---
# Lấy đường dẫn file hiện tại, đi lùi ra 2 cấp (Modules/Segments -> Root)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

# --- 2. CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG ---
# Folder dữ liệu đã được tạo bởi split_data.py
FINAL_DATASET_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset")
TRAIN_DIR = os.path.join(FINAL_DATASET_DIR, "train")
VAL_DIR = os.path.join(FINAL_DATASET_DIR, "val")

# Folder output (Lưu checkpoint và model final ngay tại thư mục gốc dự án)
OUTPUT_CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "segformer_house_output")
FINAL_MODEL_DIR = os.path.join(config.PROJECT_ROOT, "segformer_house_final")

# Kiểm tra an toàn trước khi chạy
if not os.path.exists(TRAIN_DIR):
    print("❌ LỖI: Không tìm thấy thư mục Train!")
    print("👉 Bạn đã chạy file 'Modules/Preprocess/split_data.py' chưa?")
    exit()

# --- 3. CẤU HÌNH CLASS ---
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
label2id = {v: k for k, v in id2label.items()}
NUM_CLASSES = len(id2label)

MODEL_CHECKPOINT = "nvidia/mit-b1" 

# --- 4. DATASET CLASS ---
class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        
        # Lấy danh sách ảnh, bỏ qua file ẩn
        self.images = sorted([f for f in os.listdir(self.images_dir) if not f.startswith('.')])
        self.masks_map = {f: f for f in os.listdir(self.masks_dir)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.images_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        
        # Logic tìm mask thông minh
        mask_name = img_name 
        if mask_name not in self.masks_map:
            mask_stem = os.path.splitext(img_name)[0]
            mask_name = mask_stem + ".png"
            
        segmentation_map = Image.open(os.path.join(self.masks_dir, mask_name))

        inputs = self.processor(
            images=image, 
            segmentation_maps=segmentation_map, 
            return_tensors="pt"
        )
        
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs

# --- 5. CHUẨN BỊ DỮ LIỆU ---
processor = SegformerImageProcessor.from_pretrained(
    MODEL_CHECKPOINT, 
    do_reduce_labels=False
)

train_dataset = SemanticSegmentationDataset(TRAIN_DIR, processor)
val_dataset = SemanticSegmentationDataset(VAL_DIR, processor)

# Kiểm tra nhanh
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# --- 6. KHỞI TẠO MODEL ---
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# --- 7. METRIC (IoU) ---
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = torch.nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    
    pred_labels = logits_tensor.detach().cpu().numpy()
    
    metrics = metric.compute(
        predictions=pred_labels, 
        references=labels, 
        num_labels=NUM_CLASSES, 
        ignore_index=255,
        reduce_labels=False
    )
    
    return {
        "mean_iou": metrics["mean_iou"],
        "mean_accuracy": metrics["mean_accuracy"],
        "overall_accuracy": metrics["overall_accuracy"],
        # An toàn hơn: dùng get() để tránh lỗi index nếu dataset thiếu class
        "iou_building": metrics["per_category_iou"][1] if len(metrics["per_category_iou"]) > 1 else 0.0
    }

# --- 8. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_CHECKPOINT_DIR, # Dùng đường dẫn từ config
    
    learning_rate=6e-5,          
    num_train_epochs=100,        
    lr_scheduler_type="cosine",  # <--- Thay đổi: Giảm LR theo hình sin (tốt hơn linear mặc định)
    warmup_ratio=0.1,            # <--- 10% thời gian đầu để "làm nóng" model, tránh shock
    
    # Regularization
    weight_decay=0.01,

    dataloader_num_workers=0, # Chống treo máy 
    
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, 
    per_device_eval_batch_size=4,
    
    save_total_limit=2,  # Chỉ giữ lại 2 checkpoint gần nhất
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    remove_unused_columns=False,
    push_to_hub=False,
    
    load_best_model_at_end=True,    # Train xong tự động load lại model ngon nhất
    metric_for_best_model="mean_iou", # Tiêu chí: Cái nào có Mean IoU cao nhất là NHẤT
    greater_is_better=True,
    
    fp16=False,  # Dùng FP16 nếu có GPU
)

# --- 9. BẮT ĐẦU TRAIN ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)

print("\n🚀 Bắt đầu training tiếp từ checkpoint 224...")

checkpoint_path = os.path.join(OUTPUT_CHECKPOINT_DIR, "checkpoint-224")
trainer.train(resume_from_checkpoint=checkpoint_path)

# Lưu model cuối cùng vào đường dẫn config
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(f"✅ Training hoàn tất. Model đã lưu tại: {FINAL_MODEL_DIR}")