import json
import os
import matplotlib.pyplot as plt
import sys

# Setup import config
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
import config

# Đường dẫn đến file log (Thường nằm trong folder output/checkpoint-cuối-cùng)
# Hoặc nếu train xong nó nằm ngay trong segformer_house_output
CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "segformer_house_output")

def find_latest_trainer_state(folder):
    # Tìm file trainer_state.json trong các checkpoint
    checkpoints = [d for d in os.listdir(folder) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    # Lấy checkpoint có số lớn nhất (mới nhất)
    latest_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    json_path = os.path.join(folder, latest_ckpt, "trainer_state.json")
    return json_path if os.path.exists(json_path) else None

def plot_training_history():
    json_path = find_latest_trainer_state(CHECKPOINT_DIR)
    
    if not json_path:
        print(f"❌ Không tìm thấy file 'trainer_state.json' trong {CHECKPOINT_DIR}")
        return

    print(f"📖 Đang đọc log từ: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    history = data["log_history"]
    
    epochs = []
    train_loss = []
    eval_loss = []
    eval_iou = []
    eval_acc = []

    # Tách dữ liệu
    for entry in history:
        if "loss" in entry: # Log training
            # Lưu lại epoch và loss, có thể dùng nội suy nếu cần
            pass
        
        if "eval_loss" in entry: # Log evaluation
            epochs.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])
            eval_iou.append(entry["eval_mean_iou"])
            eval_acc.append(entry["eval_accuracy"])

    # Vẽ biểu đồ
    plt.figure(figsize=(15, 5))

    # 1. Biểu đồ Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, eval_loss, label="Val Loss", color='red', marker='o')
    plt.title("Validation Loss (Càng thấp càng tốt)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 2. Biểu đồ IoU
    plt.subplot(1, 3, 2)
    plt.plot(epochs, eval_iou, label="Mean IoU", color='blue', marker='o')
    plt.title("Mean IoU (Càng cao càng tốt)")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.legend()
    
    # 3. Biểu đồ Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, eval_acc, label="Pixel Accuracy", color='green', marker='o')
    plt.title("Accuracy (Càng cao càng tốt)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_history()