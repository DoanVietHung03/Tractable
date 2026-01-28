import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Lấy đường dẫn file hiện tại, đi lùi ra 2 cấp (Modules -> Root) để tìm config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import config

def check_unique_ids(folder_path, name):
    unique_ids = set()
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
    # Lấy mẫu 10 file để kiểm tra nhanh (hoặc bỏ [:10] để check hết)
    for f in tqdm(files, desc=f"Checking {name}"):
        mask_path = os.path.join(folder_path, f)
        mask = np.array(Image.open(mask_path))
        unique_ids.update(np.unique(mask))
    print(f"--- Dataset {name} có các Class ID: {sorted(list(unique_ids))} ---")
    
if __name__ == "__main__":    
    # check IRFS
    irfs_mask_path = os.path.join(config.IRFS_DIR, "0-1-masks")
    check_unique_ids(irfs_mask_path, "IRFS")
    
    # check ETRIMS
    etrims_mask_path = os.path.join(config.ETRIMS_DIR, "masks")
    check_unique_ids(etrims_mask_path, "ETRIMS")