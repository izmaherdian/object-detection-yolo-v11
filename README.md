# Campus Object Detection with YOLO11n

Real-time detection of **cars, fountains, people and trees** with a fine-tuned **YOLO11n**. The model was trained on the Roboflow *Al-Mustansiriya University* dataset and then tested live, by webcam, on the **ITB Ganesha campus** (Bandung, Indonesia).

> Course project for **IK5127 – Artificial Intelligence for Autonomous Systems**, Engineering Physics, Institut Teknologi Bandung (2024).
> Author: **Izma Alhazmi Herdian** (13321027) · Full report: [`docs/report_yolo11_object_detection_ITB.pdf`](docs/report_yolo11_object_detection_ITB.pdf) (Indonesian)

<p align="center">
  <img src="assets/demo/realtime_detection_campus_2.gif" width="32%" alt="Real-time detection on ITB campus (1)">
  <img src="assets/demo/realtime_detection_campus_1.gif" width="32%" alt="Real-time detection on ITB campus (2)">
  <img src="assets/demo/realtime_detection_indoor.gif" width="32%" alt="Real-time detection indoors">
</p>
<p align="center"><i>Live webcam inference on ITB Ganesha campus. Full-length videos are in <a href="demo/"><code>demo/</code></a>.</i></p>

---

## Contents
- [Highlights](#highlights)
- [Demo videos](#demo-videos)
- [Method](#method)
- [Model architecture](#model-architecture-yolo11n)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Error analysis](#error-analysis)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Acknowledgements](#acknowledgements)

## Highlights

| | |
|---|---|
| **Model** | YOLO11n (Ultralytics), 2.59 M parameters, 6.4 GFLOPs |
| **Classes** | `car`, `fountain`, `person`, `tree` |
| **Data** | 755 images: 604 train / 76 val / 75 test, 17,252 boxes |
| **Training** | 20 epochs, 640 px, batch 16, AdamW (auto), ~3.2 h on a Colab CPU runtime |
| **Test mAP@0.5** | **0.562** (val 0.565) |
| **Test mAP@0.5:0.95** | **0.328** (val 0.324) |
| **Speed** | ~85–130 ms per 640×480 frame on a laptop CPU (≈ 8–12 FPS) |

## Demo videos

The trained model runs frame by frame on a laptop webcam through OpenCV ([`notebooks/03_realtime_webcam_inference.ipynb`](notebooks/03_realtime_webcam_inference.ipynb) / [`scripts/detect.py`](scripts/detect.py)):

| Video | Scene | Length |
|---|---|---|
| [`demo/realtime_detection_campus_1.mp4`](demo/realtime_detection_campus_1.mp4) | Outdoor: fountain plaza and gardens, ITB | 0:44 |
| [`demo/realtime_detection_campus_2.mp4`](demo/realtime_detection_campus_2.mp4) | Outdoor: walking around the campus | 1:39 |
| [`demo/realtime_detection_indoor.mp4`](demo/realtime_detection_indoor.mp4) | Indoor: corridors and study area | 0:29 |

Photos from the campus, detected with the same model:

![Inference on ITB photos](assets/itb_predictions.png)

## Method

![Method pipeline](assets/pipeline.png)

1. **Dataset.** We used the [Al-Mustansiriya University Data](https://universe.roboflow.com/waeel/al-mustansiriya-university-data/dataset/2) (v2) from Roboflow Universe. It has 755 campus images (drone and street-level) labelled with YOLO bounding boxes for four classes.
2. **Pre-processing (Roboflow).** Images were auto-oriented (EXIF stripped) and resized to 1280×720 (stretch). No offline augmentation was applied.
3. **Split.** The data is split 80 / 10 / 10 into train, validation and test sets.
4. **Fine-tuning (transfer learning).** Training starts from the COCO-pretrained `yolo11n.pt`. 448 of 499 weight tensors are transferred; the classification branches of the detection head are re-initialised for 4 classes. The whole network is then trained for 20 epochs at 640 px. Ultralytics' default *online* augmentations (mosaic, HSV jitter, horizontal flip, scale/translate) are applied during training. Mosaic is switched off for the last 10 epochs.
5. **Evaluation.** We report precision, recall, mAP@0.5 and mAP@0.5:0.95, together with confusion matrices and F1/PR curves on the validation and test splits.
6. **Deployment.** `best.pt` runs through OpenCV on three kinds of input: a live webcam, recorded video, and still photos.

## Model architecture (YOLO11n)

![YOLO11n architecture](assets/architecture.png)

YOLO11 is a one-stage, **anchor-free** detector with three parts:

- **Backbone.** A stack of strided `Conv` layers and **C3k2** blocks (CSP bottlenecks with two small convolutions) extracts features at strides 8, 16 and 32 (P3/P4/P5). **SPPF** pools the deepest map at several receptive-field sizes. **C2PSA** is new in YOLO11: it adds position-sensitive self-attention so the model can use global context.
- **Neck (PAN-FPN).** A top-down path upsamples deep, semantic features and merges them with shallow, high-resolution ones. A bottom-up path then passes the localisation detail back to the deeper levels.
- **Head.** Decoupled detection heads work on 80×80, 40×40 and 20×20 grids, for small, medium and large objects. Each grid cell predicts box distances with Distribution Focal Loss (DFL) and class scores directly, with no anchor boxes. Non-maximum suppression (NMS) merges the 8,400 candidates into the final detections.

The **n** (nano) variant was chosen because it is the fastest and lightest of the YOLO11 family. That matters for real-time use on a CPU-only laptop. On COCO it reaches 39.5 mAP@0.5:0.95 with 1.5 ms per image on a T4 GPU (TensorRT) and 56 ms on CPU (ONNX).

## Dataset

![Dataset statistics](assets/dataset_stats.png)

| Split | Images | car | fountain | person | tree | Total boxes |
|---|---:|---:|---:|---:|---:|---:|
| train | 604 | 1,249 | 638 | 3,086 | 8,592 | 13,565 |
| val | 76 | 110 | 77 | 351 | 1,027 | 1,565 |
| test | 75 | 285 | 72 | 325 | 1,440 | 2,122 |

The dataset is **heavily imbalanced**: trees make up about 64% of all boxes, while fountains are only about 5%. The scenes are also dense, with a median of 16 objects per image and some drone shots holding more than 150 boxes. Both points explain much of the per-class results below.

![Ground-truth samples](assets/dataset_samples.png)

<details>
<summary>Label distribution and training mosaics (generated by Ultralytics)</summary>

| Label distribution / box geometry | Label correlogram |
|---|---|
| ![labels](assets/training/labels.jpg) | ![correlogram](assets/training/labels_correlogram.jpg) |

Training batch after mosaic augmentation:

![train batch](assets/training/train_batch0.jpg)
</details>

## Training

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=20 imgsz=640
```

Loss and metric curves over the 20 epochs:

![Training curves](results/results.png)

Box, class and DFL losses are still falling at epoch 20 on both train and val, and mAP is still rising. **The model has not converged yet**, so more epochs would very likely improve accuracy.

## Results

### Overall

| Split | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Validation | 0.680 | 0.532 | 0.565 | 0.324 |
| **Test** | **0.620** | **0.505** | **0.562** | **0.328** |

### Per class

![Per-class metrics](assets/per_class_metrics.png)

| Class | Val P | Val R | Val mAP50 | Test P | Test R | Test mAP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| car | 0.569 | 0.277 | 0.277 | 0.534 | 0.260 | 0.297 | 0.106 |
| fountain | 0.675 | 0.416 | 0.437 | 0.541 | 0.431 | 0.512 | 0.268 |
| person | 0.738 | 0.709 | 0.755 | 0.703 | 0.618 | 0.672 | 0.453 |
| tree | 0.739 | 0.725 | 0.792 | 0.704 | 0.710 | 0.765 | 0.485 |

Validation and test scores are almost the same, so the model generalises consistently to unseen images from the same distribution. **Trees and people** are detected reliably. **Cars and fountains** are much weaker.

### Qualitative: ground truth vs. prediction (test split)

![GT vs prediction](assets/gt_vs_pred.png)

### Confusion matrices and curves

| Confusion matrix (val) | Normalised (val) | Normalised (test) |
|---|---|---|
| ![cm](results/confusion_matrix.png) | ![cmn](results/confusion_matrix_normalized.png) | ![cmn test](results/test/confusion_matrix_normalized.png) |

| F1–confidence | Precision–recall | Precision–confidence | Recall–confidence |
|---|---|---|---|
| ![F1](results/F1_curve.png) | ![PR](results/PR_curve.png) | ![P](results/P_curve.png) | ![R](results/R_curve.png) |

Validation curves are in [`results/`](results/) and test-split curves in [`results/test/`](results/test/).

## Error analysis

- **Missed objects are the main error, not class confusion.** In the confusion matrices, almost all off-diagonal mass is in the *background* row and column. The model rarely mistakes one class for another, but it often misses an object entirely (false negative) or fires on background (false positive).
- **Cars are the weakest class** (test recall 0.26). Most cars in the dataset are tiny, seen from a drone, and packed together. At 640 px input they cover only a few pixels, and on test 194 of 285 cars are missed.
- **Fountains** have the fewest examples (638 training boxes) and vary a lot in how they look. Only about half are found.
- **Trees vs. background.** Many *background* false positives are predicted as `tree`. Dense vegetation has no clear object boundary, so the predicted boxes often disagree with the annotated ones. That counts as both a false positive and a miss, even when the prediction is reasonable.
- **Domain shift to ITB.** The training images come from a campus in Iraq (drone shots, palm trees, desert light). On the ITB photos, trees and people transfer well, but the ITB fountain and the car in the foreground are **not** detected. Collecting a small labelled ITB dataset would help the most.

**Ways to improve:** train longer (the loss is still falling at epoch 20); use a larger input size (`imgsz=1280`) for small drone-view cars; rebalance or oversample `car` and `fountain`; add ITB-specific images; try a larger variant (YOLO11s/m) on a GPU.

## Quick start

```bash
git clone https://github.com/izmaherdian/ObjectDetectionYoloV11.git
cd ObjectDetectionYoloV11
pip install -r requirements.txt
```

**Inference** (uses `weights/best.pt`):

```bash
python scripts/detect.py --source 0                          # live webcam, press q to quit
python scripts/detect.py --source demo/realtime_detection_indoor.mp4
python scripts/detect.py --source assets/samples/ --save     # images → runs/detect/predict
```

**Python API:**

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("assets/samples/itb_ganesha_road.jpg", conf=0.25)
results[0].show()
```

**Training / evaluation:**

```bash
python scripts/train.py                     # reproduces the original run (20 epochs, 640 px)
python scripts/train.py --epochs 100 --device 0
python scripts/make_figures.py              # re-evaluates best.pt and regenerates every figure in assets/
```

## Repository structure

```
ObjectDetectionYoloV11/
├── README.md
├── requirements.txt
├── data/                         # Roboflow dataset (YOLO format)
│   ├── data.yaml                 # class names + split paths
│   ├── train/ valid/ test/       # images/ + labels/ (one .txt per image)
│   └── README.*.txt              # dataset source & license (CC BY 4.0)
├── weights/
│   └── best.pt                   # fine-tuned YOLO11n (20 epochs)
├── notebooks/
│   ├── 01_coco_subset_download.ipynb     # early exploration with COCO (not used for final model)
│   ├── 02_train_yolo11n_colab.ipynb      # original Colab training run + logs
│   └── 03_realtime_webcam_inference.ipynb
├── scripts/
│   ├── train.py                  # fine-tune + test-set evaluation
│   ├── detect.py                 # webcam / video / image inference
│   └── make_figures.py           # regenerates all README figures
├── results/                      # Ultralytics evaluation plots (val) + results/test/
├── assets/                       # README figures, GIFs, sample photos
├── demo/                         # full-length real-time detection videos
└── docs/
    ├── report_yolo11_object_detection_ITB.pdf / .docx   # project report (Indonesian)
    └── assignment_brief.docx
```

## Acknowledgements

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), the model and training framework.
- The dataset is [Al-Mustansiriya University Data](https://universe.roboflow.com/waeel/al-mustansiriya-university-data) by *waeel* on Roboflow Universe, licensed under **CC BY 4.0**.
- Course IK5127 Artificial Intelligence for Autonomous Systems, Institut Teknologi Bandung.
