import os
import pandas as pd 
# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------
# HELPER: check if folder is ImageFolder-style
# --------------------------------------------------

def is_imagefolder_root(path):
    """
    Checks if a directory follows ImageFolder structure:
    root/
        class1/
            img1.jpg
        class2/
            img2.jpg
    """
    if not os.path.isdir(path):
        return False

    subdirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]

    # Need at least 2 classes
    if len(subdirs) < 2:
        return False

    for d in subdirs:
        class_dir = os.path.join(path, d)

        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        if len(images) == 0:
            return False

    return True


# --------------------------------------------------
# STEP 1: detect pre-split dataset (train/val or train/test)
# --------------------------------------------------

def find_presplit_root(base_path):
    """
    Finds dataset root containing train + val or train + test.
    Example:
        dataset/
            train/
            val/
    """
    for root, dirs, _ in os.walk(base_path):
        dirs_lower = [d.lower() for d in dirs]

        if "train" in dirs_lower and ("val" in dirs_lower or "test" in dirs_lower):
            return root

    return None


# --------------------------------------------------
# STEP 2: detect single ImageFolder dataset
# --------------------------------------------------

def find_single_imagefolder_root(base_path):
    """
    Finds a dataset where classes are directly under one folder.
    """
    for root, _, _ in os.walk(base_path):
        if is_imagefolder_root(root):
            return root

    return None


# --------------------------------------------------
# PUBLIC API: dataset detection entry point
# --------------------------------------------------

def find_dataset_root(base_path):
    """
    Detects dataset root and structure type.

    Returns:
        root_path (str or None)
        structure_type (str or None)
            - "pre_split"
            - "single_folder"
    """

    # Priority 1: pre-split dataset
    presplit_root = find_presplit_root(base_path)
    if presplit_root:
        return presplit_root, "pre_split"

    # Priority 2: single ImageFolder
    single_root = find_single_imagefolder_root(base_path)
    if single_root:
        return single_root, "single_folder"

    # Nothing found
    return None, None


# --------------------------------------------------
# EXTRA (used later): class distribution
# --------------------------------------------------

def get_class_distribution(imagefolder_root):
    """
    Returns dict: {class_name: image_count}
    """
    distribution = {}

    for class_name in os.listdir(imagefolder_root):
        class_path = os.path.join(imagefolder_root, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        distribution[class_name] = len(images)

    return distribution


def get_recommended_details(df):
    num_classes = len(df["Class"])
    total_images = df["Image Count"].sum()
    avg_images_per_class = df["Image Count"].mean()
    min_images_per_class = df["Image Count"].min()
    max_images_per_class = df["Image Count"].max()

    data = {"num_classes":[num_classes],
            "total_images": [total_images],
            "avg_images_per_class": [int(avg_images_per_class)],
            "min_images_per_class": [min_images_per_class],
            "max_images_per_class":[max_images_per_class]
            }

    df = pd.DataFrame(data=data)
    
    
    # -------------------------------
    # 1. Backbone selection 
    # -------------------------------
    if num_classes <= 4:
        model = "mobilenetv2"
    else:
        model = "resnet18"

    # -------------------------------
    # 2. Training depth selection 
    # -------------------------------
    if avg_images_per_class < 150:
        training_approach = "Final Head Only"
        batch_size = 16
    elif avg_images_per_class < 1000:
        training_approach = "Layer4 Unfreeze + Final Head"
        batch_size = 32
    else:
        training_approach = "Layer4 Unfreeze + Final Head"
        batch_size = 64


    recommendation_table ={"model":[model],"training_approach":[training_approach],"batch_size":[batch_size],"epochs":[8]}

    return df,recommendation_table


# def generate_training_script(config):
