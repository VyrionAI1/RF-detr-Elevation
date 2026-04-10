# ============================================================
# FULL CONFUSION MATRIX CODE
# Run after coco_results is built with bbox field
# ============================================================

# ============================================================
# 1- Imports
# ============================================================
import torch
import supervision as sv
from tqdm import tqdm
from PIL import Image
from rfdetr import RFDETRSegNano
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
    pretrain_weights="checkpoint_best_total.pth",
    device=device,
    num_classes=1,
)
model.optimize_for_inference(compile=False)
print("✓ Model loaded.")

# ============================================================
# 3- Load COCO Dataset
# ============================================================
dataset_path = "/content/dataset_312"
ann_path     = f"{dataset_path}/valid/_annotations.coco.json"

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_path}/valid",
    annotations_path=ann_path,
)
print("Classes:", ds.classes)

coco_gt    = COCO(ann_path)
img_id_map = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

# ============================================================
# 4- Run Inference (RLE + bbox — pycocotools safe)
# ============================================================
coco_results = []

with torch.inference_mode():
    for i, (path, image, annotations) in enumerate(tqdm(ds)):
        image_pil = Image.open(path).convert("RGB")
        detections = model.predict(image_pil, threshold=0.0)

        fname    = path.split("/")[-1]
        image_id = img_id_map.get(fname, i + 1)

        if detections.mask is not None and len(detections.mask) > 0:
            for j in range(len(detections.mask)):
                binary_mask = detections.mask[j].astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                rle      = mask_util.encode(np.asfortranarray(binary_mask))
                rle_copy = mask_util.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox = mask_util.toBbox(rle_copy).tolist()  # [x, y, w, h]

                score = float(detections.confidence[j]) if detections.confidence is not None else 1.0
                label = int(detections.class_id[j])     if detections.class_id  is not None else 0

                coco_results.append({
                    "image_id"    : image_id,
                    "category_id" : label + 1,
                    "segmentation": rle,
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
# 6- Evaluate with pycocotools
# ============================================================
coco_dt   = coco_gt.loadRes(coco_results_filtered)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
coco_eval.evaluate()
coco_eval.accumulate()

with contextlib.redirect_stdout(io.StringIO()):
    coco_eval.summarize()

print("\n===== mAP Results =====")
print(f"mAP@50:95 : {coco_eval.stats[0]:.4f}")
print(f"mAP@50    : {coco_eval.stats[1]:.4f}")
print(f"mAP@75    : {coco_eval.stats[2]:.4f}")
print(f"mAR@100   : {coco_eval.stats[8]:.4f}")

# ============================================================
# 7- Extract TP / FP / FN (confirmed layout: img × area × cat)
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

area_all = coco_eval.params.areaRng[0]   # [0, 10000000000.0]

for img_idx in range(num_imgs):
    for area_idx in range(num_areas):
        for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
            if cat_id not in cat_id_to_idx:
                continue

            eval_idx = img_idx * num_areas * num_cats + area_idx * num_cats + cat_idx
            eval_img = coco_eval.evalImgs[eval_idx]

            if eval_img is None:
                continue
            if eval_img["aRng"] != area_all:   # only "all" area
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

print(f"\n✓ Sanity check → TP={tp_per_class}, FP={fp_per_class}, FN={fn_per_class}")

# ============================================================
# 8- Compute Precision / Recall / F1
# ============================================================
precision_per_class = np.where(
    (tp_per_class + fp_per_class) > 0,
    tp_per_class / (tp_per_class + fp_per_class), 0.0)

recall_per_class = np.where(
    (tp_per_class + fn_per_class) > 0,
    tp_per_class / (tp_per_class + fn_per_class), 0.0)

f1_per_class = np.where(
    (precision_per_class + recall_per_class) > 0,
    2 * precision_per_class * recall_per_class /
    (precision_per_class + recall_per_class), 0.0)

macro_p  = precision_per_class.mean()
macro_r  = recall_per_class.mean()
macro_f1 = f1_per_class.mean()

# ============================================================
# 9- Print Classification Report
# ============================================================
print("\n" + "="*70)
print(f"   CLASSIFICATION REPORT  (IoU ≥ {IOU_THRESH} | conf ≥ {CONF_THRESHOLD})")
print("="*70)
print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} "
      f"{'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
print("-"*70)
for i, name in enumerate(class_names):
    print(f"{name:<20} {precision_per_class[i]:>10.4f} "
          f"{recall_per_class[i]:>10.4f} {f1_per_class[i]:>10.4f} "
          f"{tp_per_class[i]:>8} {fp_per_class[i]:>8} {fn_per_class[i]:>8}")
print("-"*70)
print(f"{'macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} "
      f"{tp_per_class.sum():>8} {fp_per_class.sum():>8} {fn_per_class.sum():>8}")
print("="*70)

# ============================================================
# 10- Build Confusion Matrix
# ============================================================
size   = num_classes + 1
cm     = np.zeros((size, size), dtype=int)
labels = class_names + ["background"]

for i in range(num_classes):
    cm[i, i]           = tp_per_class[i]    # TP on diagonal
    cm[i, num_classes] = fn_per_class[i]    # FN → missed by model
    cm[num_classes, i] = fp_per_class[i]    # FP → false alarms

with np.errstate(divide="ignore", invalid="ignore"):
    cm_norm = np.where(
        cm.sum(axis=1, keepdims=True) > 0,
        cm / cm.sum(axis=1, keepdims=True), 0.0)

# ============================================================
# 11- Plot Confusion Matrix
# ============================================================
fig, ax = plt.subplots(figsize=(max(6, num_classes * 2 + 3),
                                max(5, num_classes * 2 + 2)))

sns.heatmap(
    cm_norm,
    annot=cm,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="lightgrey",
    ax=ax,
    annot_kws={"size": 13},
    vmin=0, vmax=1,
)

ax.set_xlabel("Predicted",    fontsize=13, labelpad=10)
ax.set_ylabel("Ground Truth", fontsize=13, labelpad=10)
ax.set_title(
    f"Confusion Matrix  (IoU ≥ {IOU_THRESH}  |  conf ≥ {CONF_THRESHOLD})\n"
    f"mAP@50={coco_eval.stats[1]:.3f}  |  "
    f"Precision={macro_p:.3f}   Recall={macro_r:.3f}   F1={macro_f1:.3f}",
    fontsize=13, pad=14)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
ax.legend(handles=[
    mpatches.Patch(color="#2196F3", label="TP (diagonal)"),
    mpatches.Patch(color="#90CAF9", label="FN (last col)"),
    mpatches.Patch(color="#BBDEFB", label="FP (last row)"),
], loc="upper right", fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig("/content/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
display(IPImage("/content/confusion_matrix.png"))
print("✓ Confusion matrix displayed.")

# ============================================================
# 12- Plot Per-Class Bar Chart
# ============================================================
x     = np.arange(num_classes)
width = 0.28

fig2, ax2 = plt.subplots(figsize=(max(7, num_classes * 2), 5))

b1 = ax2.bar(x - width, precision_per_class, width, label="Precision", color="#1976D2")
b2 = ax2.bar(x,          recall_per_class,   width, label="Recall",    color="#388E3C")
b3 = ax2.bar(x + width,  f1_per_class,       width, label="F1-Score",  color="#F57C00")

ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=25, ha="right")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title(
    f"Per-Class Precision / Recall / F1"
    f"  (IoU ≥ {IOU_THRESH}  |  conf ≥ {CONF_THRESHOLD})",
    fontsize=13)
ax2.axhline(y=macro_f1, color="grey", linestyle="--",
            linewidth=1.2, label=f"macro F1={macro_f1:.3f}")
ax2.legend(fontsize=10)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("/content/per_class_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
display(IPImage("/content/per_class_metrics.png"))
print("✓ Per-class bar chart displayed.")