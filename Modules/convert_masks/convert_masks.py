import cv2
import numpy as np
import os

CLASS_COLOR_GROUPS = {
    0: [(0, 0, 0)], # Background
    1: [(128, 0, 0)], # Building 
    2: [(0, 0, 128)], # Window
    3: [(128, 128, 0), (128, 64, 0)], # Door
    4: [(0, 128, 0), (170, 255, 85)], # Tree
    5: [
        (0, 128, 128),
        (0, 0, 255),
        (0, 0, 170),
        (0, 85, 255),
        (0, 170, 255)
    ], # Sky
    6: [(128, 128, 128)], # Road
    7: [(128, 0, 128)] # Car
}

COLOR_MAP = {}
for class_id, colors in CLASS_COLOR_GROUPS.items():
    for c in colors:
        COLOR_MAP[c] = class_id


def rgb_mask_to_class_id(mask_rgb):
    h, w, _ = mask_rgb.shape
    class_mask = np.full((h, w), fill_value=255, dtype=np.uint8)  # 255 = invalid

    for rgb, class_id in COLOR_MAP.items():
        r, g, b = rgb
        match = (
            (mask_rgb[:, :, 0] == r) &
            (mask_rgb[:, :, 1] == g) &
            (mask_rgb[:, :, 2] == b)
        )
        class_mask[match] = class_id

    return class_mask


def find_unknown_colors(mask_rgb):
    known = set(COLOR_MAP.keys())
    colors = set(tuple(c) for c in mask_rgb.reshape(-1, 3))
    return colors - known


input_dir  = ".\\Dataset\\etrims\\annotations"
output_dir = ".\\Dataset\\etrims\\masks"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png", ".jpg")):
        continue

    bgr = cv2.imread(os.path.join(input_dir, fname))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    unknown = find_unknown_colors(rgb)
    if len(unknown) > 0:
        print(f"[ERROR] {fname} still has unknown colors: {unknown}")
        continue  # KHÔNG convert nếu còn màu lạ

    class_mask = rgb_mask_to_class_id(rgb)

    assert 255 not in np.unique(class_mask), f"Invalid pixel in {fname}"

    cv2.imwrite(os.path.join(output_dir, fname), class_mask)
    print(f"Converted OK: {fname}")