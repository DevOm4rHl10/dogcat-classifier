import os
import shutil
import random

# Paths
source_dir = "training-data"
output_dir = "data"

# Split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Ensure reproducibility
random.seed(42)

# Classes (subfolders in source_dir)
classes = ["Cat", "Dog"]

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    images = [img for img in images if os.path.isfile(os.path.join(cls_path, img))]
    
    random.shuffle(images)
    
    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    for split, files in zip(
        ["train", "val", "test"],
        [train_files, val_files, test_files]
    ):
        split_dir = os.path.join(output_dir, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for file in files:
            src = os.path.join(cls_path, file)
            dst = os.path.join(split_dir, file)
            shutil.copy2(src, dst)

print("Dataset successfully split into train, val, and test sets!")