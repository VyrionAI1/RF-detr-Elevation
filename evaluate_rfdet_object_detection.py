# ============================================================
# RFDet Object Detection Evaluation Script
# ============================================================

# ============================================================
# 1- Imports
# ============================================================
import torch
import supervision as sv
from tqdm import tqdm
from PIL import Image
from rfdetr import RFDETRSegNano  # Assuming you use the same model to get boxes
import numpy as np
import gc
import io
import contextlib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from IPython.display import display, Image as IPImage

# ============================================================
# 2- Load Model
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RFDETRSegNano(
    pretrain_weights="checkpoint_best_total.pth", # Update path if needed
    device=device,
    num_classes=1, # Update to your number of classes
)
model.optimize_for_inference(compile=False)
print("✓ Model loaded for Object Detection.")

# ============================================================
# 3- Load COCO Dataset
# ============================================================
dataset_path = "/content/dataset_312" # Update path if needed
ann_path     = f"{dataset_path}/valid/_annotations.coco.json"

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_path}/valid",
    annotations_path=ann_path,
)
print("Classes:", ds.classes)

coco_gt    = COCO(ann_path)
img_id_map = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

# ============================================================
# 4- Run Inference (bbox only)
# ============================================================
coco_results = []

with torch.inference_mode():
    for i, (path, image, annotations) in enumerate(tqdm(ds)):
        image_pil = Image.open(path).convert("RGB")
        detections = model.predict(image_pil, threshold=0.0)

        fname    = path.split("/")[-1]
        image_id = img_id_map.get(fname, i + 1)

        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for j in range(len(detections.xyxy)):
                # [x1, y1, x2, y2] -> [x, y, w, h] for COCO
                x1, y1, x2, y2 = detections.xyxy[j]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                score = float(detections.confidence[j]) if detections.confidence is not None else 1.0
                label = int(detections.class_id[j])     if detections.class_id  is not None else 0

                coco_results.append({
                    "image_id"    : image_id,
                    "category_id" : label + 1,
                    "bbox"        : bbox,
                    "score"       : score,
                })

        del image_pil, detections, annotations
        if i % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

print(f"✓ Total detections (threshold=0.0): {len(coco_results):,}")

# ============================================================
# 5- Filter by confidence threshold
# ============================================================
CONF_THRESHOLD = 0.5   # tune: 0.3 / 0.5 / 0.7

coco_results_filtered = [r for r in coco_results if r["score"] >= CONF_THRESHOLD]
print(f"  Before filter : {len(coco_results):,}")
print(f"  After  filter : {len(coco_results_filtered):,}")

# ============================================================
# 6- Evaluate with pycocotools (iouType="bbox")
# ============================================================
if not coco_results_filtered:
    print("X No results to evaluate!")
else:
    coco_dt   = coco_gt.loadRes(coco_results_filtered)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox") # Set to bbox
    coco_eval.evaluate()
    coco_eval.accumulate()

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()

    print("\n===== mAP Results (BBox) =====")
    print(f"mAP@50:95 : {coco_eval.stats[0]:.4f}")
    print(f"mAP@50    : {coco_eval.stats[1]:.4f}")
    print(f"mAP@75    : {coco_eval.stats[2]:.4f}")
    print(f"mAR@100   : {coco_eval.stats[8]:.4f}")

    # ============================================================
    # 7- Extract TP / FP / FN 
    # ============================================================
    IOU_THRESH = 0.5
    iou_idx    = np.where(np.isclose(coco_eval.params.iouThrs, IOU_THRESH))[0][0]

    cat_ids       = sorted(coco_gt.cats.keys())
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    class_names   = [coco_gt.cats[cid]["name"] for cid in cat_ids]
    num_classes   = len(class_names)

    num_imgs  = len(coco_eval.params.imgIds)
    num_areas = len(coco_eval.params.areaRng)
    num_cats  = len(coco_eval.params.catIds)

    tp_per_class = np.zeros(num_classes, dtype=int)
    fp_per_class = np.zeros(num_classes, dtype=int)
    fn_per_class = np.zeros(num_classes, dtype=int)

    area_all = coco_eval.params.areaRng[0]   # [0, 1e10]

    for img_idx in range(num_imgs):
        for area_idx in range(num_areas):
            for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
                if cat_id not in cat_id_to_idx:
                    continue

                eval_idx = img_idx * num_areas * num_cats + area_idx * num_cats + cat_idx
                eval_img = coco_eval.evalImgs[eval_idx]

                if eval_img is None:
                    continue
                if eval_img["aRng"] != area_all: 
                    continue

                cidx       = cat_id_to_idx[cat_id]
                dt_matches = eval_img["dtMatches"]
                dt_ignore  = eval_img["dtIgnore"]
                gt_ignore  = eval_img["gtIgnore"]

                if dt_matches.shape[0] <= iou_idx:
                    continue

                matches_at_iou = dt_matches[iou_idx]
                ignore_at_iou  = dt_ignore[iou_idx].astype(bool)

                tp = int(np.sum((matches_at_iou > 0) & ~ignore_at_iou))
                fp = int(np.sum((matches_at_iou == 0) & ~ignore_at_iou))
                fn = max(0, int(np.sum(~np.array(gt_ignore, dtype=bool))) - tp)

                tp_per_class[cidx] += tp
                fp_per_class[cidx] += fp
                fn_per_class[cidx] += fn

    # ============================================================
    # 8- Compute Metrics
    # ============================================================
    precision_per_class = np.where((tp_per_class + fp_per_class) > 0, tp_per_class / (tp_per_class + fp_per_class), 0.0)
    recall_per_class = np.where((tp_per_class + fn_per_class) > 0, tp_per_class / (tp_per_class + fn_per_class), 0.0)
    f1_per_class = np.where((precision_per_class + recall_per_class) > 0, 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class), 0.0)

    macro_p  = precision_per_class.mean()
    macro_r  = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    # ============================================================
    # 9- Plot Confusion Matrix
    # ============================================================
    size   = num_classes + 1
    cm     = np.zeros((size, size), dtype=int)
    labels = class_names + ["background"]

    for i in range(num_classes):
        cm[i, i]           = tp_per_class[i]
        cm[i, num_classes] = fn_per_class[i]
        cm[num_classes, i] = fp_per_class[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0.0)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"BBox Confusion Matrix (IoU={IOU_THRESH})")
    plt.savefig("/content/bbox_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("✓ BBox confusion matrix saved.")

    # ============================================================
    # 10- Display Summary
    # ============================================================
    print("\n" + "="*30)
    print(" OBJECT DETECTION SUMMARY")
    print("="*30)
    for i, name in enumerate(class_names):
        print(f"{name:<15}: P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    print("-" * 30)
    print(f"Macro Average  : P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print("="*30)
