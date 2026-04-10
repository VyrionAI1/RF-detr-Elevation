# RFDet Instance Segmentation Evaluation

This repository provides a comprehensive evaluation suite for **RFDet (Specifically RFDETRSegNano)** instance segmentation models using the COCO dataset format. It goes beyond standard mAP metrics by providing detailed classification reports, confusion matrices, and per-class performance visualizations.

## 🚀 Features

- **Full COCO Evaluation**: Calculates mAP@50:95, mAP@50, mAP@75, and mAR@100 for both segmentation and bounding boxes.
- **Dedicated Object Detection Suite**: Separately evaluate bounding box performance (`evaluate_rfdet_object_detection.py`).
- **Detailed Classification Report**: Provides Precision, Recall, and F1-score per class at configurable IoU and confidence thresholds.
- **Confusion Matrix**: Visualizes True Positives (TP), False Positives (FP), and False Negatives (FN) including background transitions.
- **Per-Class Visualizations**: Generates bar charts for Precision, Recall, and F1-score for each category.
- **Optimized Inference**: Includes memory management (garbage collection and CUDA cache clearing) for processing large datasets.

## 🛠 Prerequisites

Ensure you have the following libraries installed:

```bash
pip install torch supervision tqdm pillow numpy matplotlib seaborn pycocotools
```

*Note: You also need the `rfdetr` library which contains the `RFDETRSegNano` architecture.*

## 📂 Project Structure

- `evealte_rfdet_instance_segmntation.py`: The main evaluation script for instance segmentation models.
- `evaluate_rfdet_object_detection.py`: A dedicated script for evaluating bounding box (object detection) performance.
- `checkpoint_best_total.pth`: (Not included) Your trained model weights.
- `dataset/`: Your COCO formatted dataset (images and `_annotations.coco.json`).

## ⚙️ Configuration

Before running the script, update the following paths in `evealte_rfdet_instance_segmntation.py`:

- **Model Weights**: Update line 38 with your path to `.pth` weights.
- **Dataset Path**: Update lines 48-49 with the path to your COCO dataset.
- **Thresholds**: You can tune `CONF_THRESHOLD` (line 107) and `IOU_THRESH` (line 133).

## 🏃 How to Run

Simply execute the Python script:

```bash
python evealte_rfdet_instance_segmntation.py
```

## 📊 Outputs

The script will generate and display:
1.  **mAP Results**: Standard COCO metrics printed to the console.
2.  **Classification Report**: A table showing per-class Precision, Recall, and F1.
3.  **Confusion Matrix**: Saved as `confusion_matrix.png`.
4.  **Per-Class Metrics**: Saved as `per_class_metrics.png`.

## 📄 License

This project is open-source and free to use.

---
*Created for evaluating RFDet Instance Segmentation models.*
