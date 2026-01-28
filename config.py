import os

# 1. Lấy đường dẫn tuyệt đối của thư mục gốc dự án (nơi chứa file config.py)
# os.path.abspath(__file__) -> Lấy đường dẫn file config.py
# os.path.dirname(...) -> Lấy thư mục chứa nó (chính là thư mục gốc HProjecT/Tractable)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. Định nghĩa các đường dẫn con dựa trên PROJECT_ROOT
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

# Đường dẫn cụ thể đến từng bộ dữ liệu
ETRIMS_DIR = os.path.join(DATASET_DIR, "others", "1-ETRIMs")
IRFS_DIR = os.path.join(DATASET_DIR, "others", "0-IRFs")

# Đường dẫn output (nếu cần)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output")

# In ra để kiểm tra khi chạy
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir:  {DATASET_DIR}")