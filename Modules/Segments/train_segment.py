import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# --- 1. SETUP CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

# --- 2. PATHS ---
FINAL_DATASET_DIR = os.path.join(config.PROJECT_ROOT, "Final_Dataset")
TRAIN_DIR = os.path.join(FINAL_DATASET_DIR, "train")
VAL_DIR = os.path.join(FINAL_DATASET_DIR, "val")
OUTPUT_CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "segformer_house_output")
FINAL_MODEL_DIR = os.path.join(config.PROJECT_ROOT, "segformer_house_final")

# --- 3. CLASS CONFIG (Để ngoài để các file khác import được) ---
id2label = {
    0: "background", 1: "building", 2: "window", 3: "door",
    4: "tree", 5: "sky", 6: "road", 7: "car"
}
label2id = {v: k for k, v in id2label.items()}
NUM_CLASSES = len(id2label)
MODEL_CHECKPOINT = "nvidia/mit-b1"

# --- 4. DATASET CLASS (Để ngoài để các file khác import được) ---
class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, processor, augment=False):
        self.root_dir = root_dir
        self.processor = processor
        self.augment = augment 
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        
        self.images = sorted([f for f in os.listdir(self.images_dir) if not f.startswith('.')])
        self.masks_map = {f: f for f in os.listdir(self.masks_dir)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.images_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        
        mask_name = img_name 
        if mask_name not in self.masks_map:
            mask_stem = os.path.splitext(img_name)[0]
            mask_name = mask_stem + ".png"
        segmentation_map = Image.open(os.path.join(self.masks_dir, mask_name))

        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                segmentation_map = segmentation_map.transpose(Image.FLIP_LEFT_RIGHT)
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        inputs = self.processor(
            images=image, 
            segmentation_maps=segmentation_map, 
            return_tensors="pt"
        )
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs

# --- 5. HÀM METRIC ---
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = torch.nn.functional.interpolate(
        logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False,
    ).argmax(dim=1)
    
    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels, references=labels, 
        num_labels=NUM_CLASSES, ignore_index=255, reduce_labels=False
    )
    
    per_category_iou = metrics.pop("per_category_iou")
    # Xử lý NaN thành 0.0 để tránh lỗi
    per_category_iou = [0.0 if np.isnan(x) else x for x in per_category_iou]
    
    results = {
        "mean_iou": metrics["mean_iou"],
        "accuracy": metrics["overall_accuracy"],
    }
    if len(per_category_iou) > 4: 
        results["iou_building"] = per_category_iou[1]
        results["iou_window"] = per_category_iou[2]
        results["iou_tree"] = per_category_iou[4]
        
    return results

# ==============================================================================
# PHẦN CHÍNH: CHỈ CHẠY KHI BẠN GÕ LỆNH "python train_segment.py"
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(TRAIN_DIR):
        print("❌ LỖI: Không tìm thấy thư mục Train!")
        exit()

    print("⏳ Đang chuẩn bị dữ liệu và model...")
    
    # 1. Prepare Data
    processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT, do_reduce_labels=False)
    train_dataset = SemanticSegmentationDataset(TRAIN_DIR, processor, augment=True)
    val_dataset = SemanticSegmentationDataset(VAL_DIR, processor, augment=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 2. Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # 3. Training Args (Cấu hình cho RTX 3060)
    training_args = TrainingArguments(
        output_dir=OUTPUT_CHECKPOINT_DIR,
        learning_rate=6e-5,
        num_train_epochs=50,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        dataloader_num_workers=8, # Tận dụng 8 core CPU
        per_device_train_batch_size=16, # Batch size lớn cho RTX 3060
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=16,
        
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        
        fp16=True, # Bật tăng tốc FP16
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    print("\n🚀 Bắt đầu training...")
    trainer.train()

    trainer.save_model(FINAL_MODEL_DIR)
    processor.save_pretrained(FINAL_MODEL_DIR)
    print(f"✅ Training hoàn tất. Model lưu tại: {FINAL_MODEL_DIR}")