import os
import random
import shutil

img_train = "dataset/yolo_human/images/train"
lbl_train = "dataset/yolo_human/labels/train"

img_val = "dataset/yolo_human/images/val"
lbl_val = "dataset/yolo_human/labels/val"

os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

images = [f for f in os.listdir(img_train)
          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

val_count = max(1, int(0.2 * len(images)))
val_images = random.sample(images, val_count)

for img in val_images:
    shutil.move(os.path.join(img_train, img), img_val)

    label = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(lbl_train, label)

    if os.path.exists(label_path):
        shutil.move(label_path, lbl_val)

print(f"✅ Moved {len(val_images)} images to validation set")
