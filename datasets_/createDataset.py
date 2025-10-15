import os
import json
import shutil
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO

CFG='configs/config.yaml'

def create_filtered_coco(src_root, dst_root, config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    categories = set(cfg["dataset"]["categories"])
    require_all = cfg["dataset"].get("require_all", True)  # default: must contain all categories

    # Create dirs
    os.makedirs(dst_root, exist_ok=True)
    os.makedirs(os.path.join(dst_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "val"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "annotations"), exist_ok=True)

    for split in ["train", "val"]:
        print(f"Processing {split}...")
        ann_file = os.path.join(src_root, "annotations", f"instances_{split}2017.json")
        coco = COCO(ann_file)

        # Filter annotations
        print("Analyzing dataset...")
        img_to_cats = {}
        for ann in tqdm(coco.dataset["annotations"]):
            img_to_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

        # Keep only image_ids that contain ALL desired categories
        print("Filtering images...")
        print("Filtering images...")
        if require_all:
            valid_img_ids = { img_id for img_id, cats in img_to_cats.items() if categories.issubset(cats) }
            print(f"Keeping only images containing ALL categories: {categories}")
        else:
            valid_img_ids = { img_id for img_id, cats in img_to_cats.items() if len(categories.intersection(cats)) > 0 }
            print(f"Keeping images containing ANY of categories: {categories}")

        anns = [ann for ann in coco.dataset["annotations"] if ann["image_id"] in valid_img_ids]
        imgs = [img for img in coco.dataset["images"] if img["id"] in valid_img_ids]
        cats = [cat for cat in coco.dataset["categories"] if cat["id"] in categories]

        print(f"Found {len(imgs)} valid images and {len(anns)} annotations in {split} set")

        # Keep only relevant images
        print(f"Copying {split} images...")
        src_img_root = os.path.join(src_root, split)
        dst_img_root = os.path.join(dst_root, split)
        for img in tqdm(imgs):
            src_img = os.path.join(src_img_root, img["file_name"])
            dst_img = os.path.join(dst_img_root, img["file_name"])
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)
        print(f"✅ Done with {split} split → {len(imgs)} images copied.")

        # Save filtered annotations
        filtered_json = {
            "images": imgs,
            "annotations": anns,
            "categories": cats
        }

        out_ann_file = os.path.join(dst_root, "annotations", f"instances_{split}.json")
        with open(out_ann_file, "w") as f:
            json.dump(filtered_json, f)
        print(f"Saved filtered JSON → {out_ann_file}")


if __name__ == "__main__":
    src_root = "data/coco2017"
    dst_root = "data/coco_person_ball"

    create_filtered_coco(src_root, dst_root, CFG)
