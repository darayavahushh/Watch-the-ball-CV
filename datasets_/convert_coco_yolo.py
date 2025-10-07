import os
import shutil
import json
from tqdm import tqdm
from ultralytics.data.converter import convert_coco

ANN_COCO = 'data/coco_person_ball/annotations'
CLEAN_ANN = 'data/coco_person_ball/annotations_cleaned'
SAVE_YOLO = 'data/coco_person_ball/yolo_coco'
TARGET = 'data/coco_person_ball'
LABELS = 'data/coco_person_ball/labels'

# COCO person=1, sports ball=37 → YOLO person=0, sports ball=1
class_mapping = {1: 0, 37: 1}


def remap_and_clean_coco(src_json, dst_json):
    """
    Clean and remap COCO annotations for YOLO training.
    Keeps only annotations in class_mapping and remaps them.

    Args:
        src_json (str): Path to source COCO JSON file.
        dst_json (str): Destination JSON file.
    """
    print(f"\nLoading {src_json}...")
    with open(src_json, "r") as f:
        coco_data = json.load(f)

    anns = coco_data["annotations"]
    imgs = coco_data["images"]
    cats = coco_data["categories"]

    kept_annotations = []
    valid_image_ids = set()

    # Keep only desired category IDs and remap them
    for ann in tqdm(anns, desc="Filtering annotations"):
        cat_id = ann["category_id"]
        bbox = ann["bbox"]

        # Skip if not in mapping or bbox invalid
        if cat_id not in class_mapping:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or any(v is None for v in bbox):
            continue

        # Create a clean copy of the annotation
        new_ann = ann.copy()
        new_ann["category_id"] = class_mapping[cat_id]
        kept_annotations.append(new_ann)
        valid_image_ids.add(new_ann["image_id"])

    # Keep only the images that contain valid annotations
    kept_images = [img for img in imgs if img["id"] in valid_image_ids]

    # Create clean category entries for YOLO
    id_to_name = {cat["id"]: cat["name"] for cat in cats}
    kept_categories = [
        {"id": new_id, "name": id_to_name[old_id]}
        for old_id, new_id in class_mapping.items()
        if old_id in id_to_name
    ]

    # Construct new dataset structure
    filtered_json = {
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": kept_categories
    }

    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(filtered_json, f)

    print("\nCleaned dataset saved:")
    print(f"  Images kept: {len(kept_images)}")
    print(f"  Annotations kept: {len(kept_annotations)}")
    print(f"  Categories: {[c['name'] for c in kept_categories]}")
    print(f"  → {dst_json}")



def relocate_labels(converted_dir, target_root):
    """
    Moves labels/ folder up two levels and removes the temporary converted directory.

    Args:
        converted_dir: path to folder created by convert_coco (e.g. ".../filtered_person_ball/coco_yolo")
        target_root: root path where images/ already exist (e.g. ".../filtered_person_ball")
    """
    labels_src = os.path.join(converted_dir, "labels")
    labels_dst = os.path.join(target_root, "labels")

    if not os.path.exists(labels_src):
        raise FileNotFoundError(f"Labels folder not found at {labels_src}")

    # Move labels folder
    if os.path.exists(labels_dst):
        shutil.rmtree(labels_dst)  # Remove old labels if present
    shutil.move(labels_src, labels_dst)

    # Delete temporary conversion directory
    shutil.rmtree(converted_dir)
    print(f"Moved {labels_src} to {labels_dst} and removed {converted_dir}")


if __name__ == "__main__":
    os.makedirs(CLEAN_ANN, exist_ok=True)
    for split in ["train", "val"]:
        src_json = os.path.join(ANN_COCO, f"instances_{split}.json")
        dst_json = os.path.join(CLEAN_ANN, f"{split}.json")
        remap_and_clean_coco(src_json, dst_json)

    print("\nRemapping done → proceeding with conversion...\n")

    convert_coco(
        labels_dir=CLEAN_ANN,
        save_dir=SAVE_YOLO,
        use_segments=False,  # bounding boxes only
        cls91to80=False,     # use class IDs from JSON
        # remove the -1 from line 310 from convert.py for: ann["category_id"]
    )
    # Relocate the output
    relocate_labels(SAVE_YOLO, TARGET)
