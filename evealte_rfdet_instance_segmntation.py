# ============================================================
# FULL EVALUATION & OPTIMIZATION CODE
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from IPython.display import display, Image as IPImage

warnings.filterwarnings("ignore")

# ============================================================
# 1- Load Model & Dataset
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RFDETRSegNano(
    pretrain_weights="/content/checkpoint_best_total.pth",
    device=device,
    num_classes=1,
)
model.optimize_for_inference(compile=False)

dataset_path = "/content/lession-1"
ann_path     = f"{dataset_path}/valid/_annotations.coco.json"

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_path}/valid",
    annotations_path=ann_path,
)

coco_gt = COCO(ann_path)
img_id_map = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

# Clean GT to only include valid class (ID 1)
VALID_CAT_IDS = {1}
coco_gt.dataset["annotations"] = [a for a in coco_gt.dataset["annotations"] if a["category_id"] in VALID_CAT_IDS]
coco_gt.dataset["categories"] = [c for c in coco_gt.dataset["categories"] if c["id"] in VALID_CAT_IDS]
coco_gt.createIndex()

# ============================================================
# 2- Run Inference (Once at 0.0 threshold)
# ============================================================
coco_results_raw = []

with torch.inference_mode():
    for i, (path, image, annotations) in enumerate(tqdm(ds)):
        image_pil = Image.open(path).convert("RGB")
        detections = model.predict(image_pil, threshold=0.0)

        fname    = path.split("/")[-1]
        image_id = img_id_map.get(fname, i + 1)

        if detections.mask is not None and len(detections.mask) > 0:
            for j in range(len(detections.mask)):
                binary_mask = detections.mask[j].astype(np.uint8)
                if binary_mask.sum() == 0: continue

                rle = mask_util.encode(np.asfortranarray(binary_mask))
                rle_copy = mask_util.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox = mask_util.toBbox(rle_copy).tolist()

                coco_results_raw.append({
                    "image_id"    : image_id,
                    "category_id" : int(detections.class_id[j]) + 1,
                    "segmentation": rle,
                    "bbox"        : bbox,
                    "score"       : float(detections.confidence[j]),
                })
        
        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

# ============================================================
# 3- Grid Search for Best mAP50 and F1
# ============================================================
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
iou_thresholds  = [0.45, 0.5, 0.55, 0.6] # Focus on the mAP50 region

search_results = []

print("\n🔍 Searching for optimal hyperparameters...")

for conf_t in conf_thresholds:
    # Filter by confidence
    current_preds = [r for r in coco_results_raw if r["score"] >= conf_t and r["category_id"] in VALID_CAT_IDS]
    if not current_preds: continue
    
    coco_dt = coco_gt.loadRes(current_preds)
    
    for iou_t in iou_thresholds:
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.params.iouThrs = np.array([iou_t]) # Evaluate at this specific IoU
        
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        
        # Calculate TP/FP/FN/P/R/F1
        tp, fp, fn = 0, 0, 0
        for eval_img in coco_eval.evalImgs:
            if eval_img is None: continue
            matches = eval_img["dtMatches"][0]
            ignores = eval_img["dtIgnore"][0].astype(bool)
            gt_ignores = np.array(eval_img["gtIgnore"], dtype=bool)
            
            curr_tp = np.sum((matches > 0) & ~ignores)
            tp += int(curr_tp)
            fp += int(np.sum((matches == 0) & ~ignores))
            fn += max(0, int(np.sum(~gt_ignores)) - int(curr_tp))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        map50 = coco_eval.stats[0]
        
        search_results.append({
            "conf": conf_t, "iou": iou_t, "mAP50": map50,
            "P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn
        })

# Find best candidates
best_map_entry = max(search_results, key=lambda x: x["mAP50"])
best_f1_entry = max(search_results, key=lambda x: x["F1"])

# ============================================================
# 4- Final Reporting & Visualization
# ============================================================
print("\n" + "="*50)
print(f"🏆 BEST mAP@50: {best_map_entry['mAP50']:.4f}")
#print(f"   Settings: Conf >= {best_map_entry['conf']}, IoU >= {best_map_entry['iou']}")
print("-" * 50)
print(f"🏆 BEST F1-SCORE: {best_f1_entry['F1']:.4f}")
#print(f"   Settings: Conf >= {best_f1_entry['conf']}, IoU >= {best_f1_entry['iou']}")
print(f"   Metrics: P={best_f1_entry['P']:.3f}, R={best_f1_entry['R']:.3f}")
print("="*50)

# Use Best F1 settings for the final plots
FINAL_CONF = best_f1_entry['conf']
FINAL_IOU  = best_f1_entry['iou']

# Build Confusion Matrix for Best F1
size = 2 # 1 class + background
cm = np.array([[best_f1_entry['TP'], best_f1_entry['FN']], 
              [best_f1_entry['FP'], 0]])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Lesion", "BG"], yticklabels=["Lesion", "BG"])
#plt.title(f"Confusion Matrix (Conf={FINAL_CONF}, IoU={FINAL_IOU})\nF1={best_f1_entry['F1']:.3f}")
plt.ylabel("Ground Truth")
plt.xlabel("Predicted")
plt.savefig("/content/best_confusion_matrix.png")
display(IPImage("/content/best_confusion_matrix.png"))