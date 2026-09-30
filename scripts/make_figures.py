"""Generate every figure used in the README (saved to assets/).

    python scripts/make_figures.py

Figures:
    architecture.png       YOLO11n backbone / neck / head diagram
    pipeline.png           end-to-end method pipeline
    dataset_stats.png      images per split, instances per class, objects per image
    dataset_samples.png    ground-truth annotations on training images
    per_class_metrics.png  mAP50 / mAP50-95 per class on the val and test splits
    gt_vs_pred.png         ground truth vs. model prediction on test images
    itb_predictions.png    predictions on photos taken at ITB Ganesha campus
It also copies the test-split evaluation plots into results/test/.
"""
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
WEIGHTS = ROOT / "weights" / "best.pt"
CLASSES = ["car", "fountain", "person", "tree"]
SPLITS = {"train": "train", "val": "valid", "test": "test"}

# Validated categorical palette (identity per class / per split), light surface
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {"car": "#2a78d6", "fountain": "#eb6834", "person": "#1baf7a", "tree": "#eda100"}
SPLIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

DEFAULT_RC = {k: v for k, v in plt.rcParams.items() if k != "backend"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.axisbelow": True,
})


def hex_to_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def read_labels(split):
    out = {}
    for lbl in sorted((DATA / split / "labels").glob("*.txt")):
        rows = [r.split() for r in lbl.read_text().strip().splitlines() if r.strip()]
        out[lbl.stem] = [(int(r[0]), *map(float, r[1:5])) for r in rows]
    return out


def image_for(split, stem):
    return DATA / split / "images" / f"{stem}.jpg"


def draw_boxes(img, boxes, labels_xywhn=True, confs=None, thickness=3):
    h, w = img.shape[:2]
    for i, b in enumerate(boxes):
        c, x, y, bw, bh = b
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        color = hex_to_bgr(CLASS_COLORS[CLASSES[c]])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = CLASSES[c] + (f" {confs[i]:.2f}" if confs is not None else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, max(th + 10, y1)), color, -1)
        cv2.putText(img, text, (x1 + 4, max(th + 4, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def legend_handles():
    return [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASSES]


# --------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc, ec="#2b2b2b", fs=10, sub=None, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fs, weight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 2, color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold", color=tc)


def arrow(ax, p, q, color="#6b6a66", style="-|>", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(15, 9.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    ax.grid(False)

    C_CONV, C_C3K2, C_SPPF, C_PSA, C_UP, C_CAT, C_DET = (
        "#cde2fb", "#fde3cf", "#d9f2e6", "#e9e4fb", "#f0efec", "#f0efec", "#fbe1e1")

    # column panels
    for x, title in [(0.2, "BACKBONE  (feature extraction)"), (5.0, "NECK  (PAN-FPN multi-scale fusion)"),
                     (11.2, "HEAD  (anchor-free detection)")]:
        w = 4.4 if x < 5 else (5.8 if x < 11 else 3.6)
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 8.5, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc="#f6f5f2", ec=GRID, lw=1))
        ax.text(x + w / 2, 8.45, title, ha="center", fontsize=11, weight="bold", color=INK2)

    ax.text(7.5, 9.1, "YOLO11n architecture  ·  2.59 M parameters  ·  6.4 GFLOPs  ·  input 640×640×3",
            ha="center", fontsize=14, weight="bold")

    # backbone (bottom -> top)
    bb = [("Input", "640×640×3", "#ffffff"), ("0 Conv 3×3/2", "320×320×16", C_CONV),
          ("1 Conv 3×3/2", "160×160×32", C_CONV), ("2 C3k2", "160×160×64", C_C3K2),
          ("3 Conv 3×3/2", "80×80×64", C_CONV), ("4 C3k2  → P3", "80×80×128", C_C3K2),
          ("5 Conv 3×3/2", "40×40×128", C_CONV), ("6 C3k2  → P4", "40×40×128", C_C3K2),
          ("7 Conv 3×3/2", "20×20×256", C_CONV), ("8 C3k2", "20×20×256", C_C3K2),
          ("9 SPPF", "20×20×256", C_SPPF), ("10 C2PSA → P5", "20×20×256", C_PSA)]
    bw, bh, bx = 3.0, 0.56, 0.9
    ys = [0.4 + i * 0.67 for i in range(len(bb))]
    ys = [y * 0.985 for y in ys]
    pos = {}
    for (t, s, c), y in zip(bb, ys):
        box(ax, bx, y, bw, bh, t, c, sub=s, fs=9.5)
    for i in range(len(bb) - 1):
        arrow(ax, (bx + bw / 2, ys[i] + bh), (bx + bw / 2, ys[i + 1]))
    pos["P3"], pos["P4"], pos["P5"] = ys[5], ys[7], ys[11]

    # neck: top-down (right-hand column at x=5.4) then bottom-up (x=8.3)
    nx1, nx2, nw = 5.35, 8.35, 2.3
    td = [("11 Upsample ×2", "40×40", C_UP, 7.25), ("12 Concat (+P4)", "40×40", C_CAT, 6.45),
          ("13 C3k2", "40×40×128", C_C3K2, 5.65), ("14 Upsample ×2", "80×80", C_UP, 4.35),
          ("15 Concat (+P3)", "80×80", C_CAT, 3.55), ("16 C3k2", "80×80×64", C_C3K2, 2.75)]
    for t, s, c, y in td:
        box(ax, nx1, y, nw, bh, t, c, sub=s, fs=9.5)
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        arrow(ax, (nx1 + nw / 2, td[a][3]), (nx1 + nw / 2, td[b][3] + bh))
    arrow(ax, (nx1 + nw / 2, td[2][3]), (nx1 + nw / 2, td[3][3] + bh))
    bu = [("17 Conv 3×3/2", "40×40×64", C_CONV, 2.75), ("18 Concat (+13)", "40×40", C_CAT, 3.55),
          ("19 C3k2", "40×40×128", C_C3K2, 4.35), ("20 Conv 3×3/2", "20×20×128", C_CONV, 5.65),
          ("21 Concat (+10)", "20×20", C_CAT, 6.45), ("22 C3k2", "20×20×256", C_C3K2, 7.25)]
    for t, s, c, y in bu:
        box(ax, nx2, y, nw, bh, t, c, sub=s, fs=9.5)
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        arrow(ax, (nx2 + nw / 2, bu[a][3] + bh), (nx2 + nw / 2, bu[b][3]))
    arrow(ax, (nx2 + nw / 2, bu[2][3] + bh), (nx2 + nw / 2, bu[3][3]))

    # skip connections backbone -> neck
    arrow(ax, (bx + bw, pos["P5"] + bh / 2), (nx1, td[0][3] + bh / 2), color="#2a78d6", lw=1.8)
    arrow(ax, (bx + bw, pos["P4"] + bh / 2), (nx1, td[1][3] + bh / 2), color="#2a78d6", lw=1.8)
    arrow(ax, (bx + bw, pos["P3"] + bh / 2), (nx1, td[4][3] + bh / 2), color="#2a78d6", lw=1.8)
    arrow(ax, (nx1 + nw, td[5][3] + bh / 2), (nx2, bu[0][3] + bh / 2), color="#6b6a66")
    arrow(ax, (nx1 + nw, td[2][3] + bh / 2), (nx2, bu[1][3] + bh / 2), color="#eb6834", lw=1.6)
    arrow(ax, (bx + bw, pos["P5"] + bh * 0.85), (nx2, bu[4][3] + bh * 0.8), color="#eb6834", lw=1.6, rad=-0.2)

    # head
    hx, hw = 11.55, 2.9
    heads = [("Detect P3/8", "80×80 grid · small objects", 2.0, td[5][3], nx1 + nw),
             ("Detect P4/16", "40×40 grid · medium objects", 4.35, bu[2][3], nx2 + nw),
             ("Detect P5/32", "20×20 grid · large objects", 7.25, bu[5][3], nx2 + nw)]
    for t, s, y, src_y, src_x in heads:
        box(ax, hx, y, hw, bh + 0.05, t, C_DET, sub=s, fs=9.5)
        start = (src_x, src_y + bh / 2) if src_x > 8 else (nx1 + nw / 2, src_y)
        arrow(ax, start, (hx, y + bh / 2), color="#e34948", lw=1.6, rad=0.0 if src_x > 8 else 0.12)
    ax.text(hx + hw / 2, 1.35, "each scale → box regression (DFL, 4×16)\n+ class scores (4 classes)",
            ha="center", fontsize=9.5, color=INK2)
    box(ax, hx, 0.35, hw, 0.7, "NMS → final boxes", "#ffffff", sub="car · fountain · person · tree", fs=10)
    ax.text(hx + hw / 2, 5.5, "8,400 candidate boxes\n(6400 + 1600 + 400)", ha="center", fontsize=9.5, color=INK2)

    # legend of block types
    lg = [(C_CONV, "Conv = Conv2d + BN + SiLU"), (C_C3K2, "C3k2 = CSP bottleneck, k=2 conv"),
          (C_SPPF, "SPPF = spatial pyramid pooling (fast)"), (C_PSA, "C2PSA = CSP + position-sensitive attention")]
    for (c, t), x in zip(lg, [0.2, 3.35, 6.85, 10.75]):
        ax.add_patch(FancyBboxPatch((x, -0.35), 0.35, 0.28, boxstyle="round,pad=0.01", fc=c, ec="#2b2b2b", lw=1))
        ax.text(x + 0.5, -0.21, t, va="center", fontsize=9.5, color=INK2)
    ax.set_ylim(-0.5, 9.4)
    fig.savefig(ASSETS / "architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(17, 4.6))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.grid(False)
    steps = [
        ("1  Dataset", "Roboflow Universe\nAl-Mustansiriya Univ. v2\n755 images · 4 classes", "#cde2fb"),
        ("2  Preprocess", "auto-orient (EXIF)\nresize 1280×720 (stretch)\nno augmentation", "#fde3cf"),
        ("3  Split", "train 604 (80%)\nvalid 76 (10%)\ntest 75 (10%)", "#d9f2e6"),
        ("4  Fine-tune", "YOLO11n (COCO-pretrained)\n20 epochs · 640 px · batch 16\nGoogle Colab", "#e9e4fb"),
        ("5  Evaluate", "P · R · mAP50 · mAP50-95\nconfusion matrix\nF1 / PR curves", "#fbe1e1"),
        ("6  Deploy", "best.pt + OpenCV\nwebcam · video · photo\nITB Ganesha campus", "#fff2cc"),
    ]
    w, gap = 2.45, 0.28
    for i, (t, s, c) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.9), w, 2.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=c, ec="#2b2b2b", lw=1.2))
        ax.text(x + w / 2, 3.3, t, ha="center", va="center", fontsize=12.5, weight="bold")
        ax.text(x + w / 2, 2.05, s, ha="center", va="center", fontsize=9.5, color=INK2, linespacing=1.6)
        if i < len(steps) - 1:
            arrow(ax, (x + w, 2.3), (x + w + gap, 2.3), color=INK, lw=1.8)
    ax.text(8.3, 4.25, "Method pipeline", ha="center", fontsize=15, weight="bold")
    ax.text(8.3, 0.45, "Transfer learning: the COCO-pretrained backbone is kept; the detection head is re-initialised "
                       "for 4 classes and the whole network is fine-tuned.", ha="center", fontsize=10.5, color=INK2)
    fig.savefig(ASSETS / "pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- dataset
def fig_dataset_stats(labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [0.8, 1.6, 1.3]})

    ax = axes[0]
    names = list(SPLITS)
    n_img = [len(list((DATA / SPLITS[s] / "images").glob("*.jpg"))) for s in names]
    bars = ax.bar(names, n_img, width=0.5, color="#2a78d6")
    for b, v in zip(bars, n_img):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=11, color=INK)
    ax.set_title("Images per split")
    ax.set_ylabel("images")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    x = np.arange(len(CLASSES))
    wdt = 0.26
    for j, s in enumerate(names):
        cnt = Counter(b[0] for boxes in labels[s].values() for b in boxes)
        vals = [cnt.get(i, 0) for i in range(len(CLASSES))]
        bars = ax.bar(x + (j - 1) * (wdt + 0.02), vals, width=wdt, color=SPLIT_COLORS[j], label=s)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, CLASSES)
    ax.set_title("Annotated instances per class")
    ax.set_ylabel("bounding boxes")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    per_img = [len(bx) for s in names for bx in labels[s].values()]
    ax.hist(per_img, bins=range(0, max(per_img) + 3, 2), color="#2a78d6", rwidth=0.85)
    ax.axvline(np.median(per_img), color=INK, lw=1.5)
    ax.text(np.median(per_img) + 30, ax.get_ylim()[1] * 0.85, f"median {np.median(per_img):.0f} boxes/image",
            fontsize=10, color=INK2)
    ax.set_title("Objects per image (all splits)")
    ax.set_xlabel("boxes in image")
    ax.set_ylabel("images")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(ASSETS / "dataset_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_samples(labels):
    rng = random.Random(7)
    # prefer images that show each class at least once
    picks = []
    for c in [0, 1, 0, 1, 2, 3]:
        cands = [k for k, v in labels["train"].items() if any(b[0] == c for b in v) and k not in picks]
        picks.append(rng.choice(cands))
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.4))
    for ax, stem in zip(axes.flat, picks):
        img = cv2.imread(str(image_for("train", stem)))
        img = draw_boxes(img, labels["train"][stem])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.legend(legend_handles(), CLASSES, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.suptitle("Ground-truth annotations (training split)", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=1.5)
    fig.savefig(ASSETS / "dataset_samples.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- results
def evaluate(model):
    res = {}
    for split in ["val", "test"]:
        with plt.rc_context(DEFAULT_RC):  # keep ultralytics' own plot style
            m = model.val(data=str(DATA / "data.yaml"), split=split, device="cpu", verbose=False,
                          project=str(ROOT / "runs" / "eval"), name=split, exist_ok=True)
        res[split] = {
            "map50": list(m.box.ap50), "map": list(m.box.ap), "p": list(m.box.p), "r": list(m.box.r),
            "all": (m.box.mp, m.box.mr, m.box.map50, m.box.map),
        }
    # keep the test-split plots alongside the original (val) ones
    dst = ROOT / "results" / "test"
    dst.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "runs" / "eval" / "test").glob("*.png"):
        shutil.copy(f, dst / f.name)
    return res


def fig_per_class(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    x = np.arange(len(CLASSES) + 1)
    labels_x = CLASSES + ["all"]
    for ax, key, title in [(axes[0], "map50", "mAP@0.5"), (axes[1], "map", "mAP@0.5:0.95")]:
        for j, split in enumerate(["val", "test"]):
            vals = res[split][key] + [res[split]["all"][2 if key == "map50" else 3]]
            bars = ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=SPLIT_COLORS[j], label=split)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
        ax.set_xticks(x, labels_x)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("average precision")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Per-class detection accuracy (best.pt, 20 epochs)", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gt_vs_pred(model, labels):
    stems = sorted(labels["test"], key=lambda k: -len({b[0] for b in labels["test"][k]}))
    rng = random.Random(3)
    picks = rng.sample(stems[:25], 3)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
    for row, stem in enumerate(picks):
        path = image_for("test", stem)
        gt = draw_boxes(cv2.imread(str(path)), labels["test"][stem])
        r = model.predict(str(path), conf=0.25, verbose=False, device="cpu")[0]
        boxes = [(int(c), *xywh) for c, xywh in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        pred = draw_boxes(cv2.imread(str(path)), boxes, confs=r.boxes.conf.tolist())
        for col, (im, t) in enumerate([(gt, f"Ground truth ({len(labels['test'][stem])} boxes)"),
                                       (pred, f"YOLO11n prediction ({len(boxes)} boxes, conf ≥ 0.25)")]):
            axes[row, col].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(t, fontsize=12)
            axes[row, col].axis("off")
    fig.legend(legend_handles(), CLASSES, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.suptitle("Ground truth vs. prediction on unseen test images", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(ASSETS / "gt_vs_pred.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def fig_itb(model):
    imgs = sorted((ASSETS / "samples").glob("*.jpg"))
    fig, axes = plt.subplots(1, len(imgs), figsize=(16, 6.5), gridspec_kw={"width_ratios": [
        cv2.imread(str(p)).shape[1] / cv2.imread(str(p)).shape[0] for p in imgs]})
    for ax, p in zip(np.atleast_1d(axes), imgs):
        r = model.predict(str(p), conf=0.25, verbose=False, device="cpu")[0]
        boxes = [(int(c), *xywh) for c, xywh in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        im = draw_boxes(cv2.imread(str(p)), boxes, confs=r.boxes.conf.tolist(), thickness=3)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(p.stem.replace("_", " "), fontsize=12)
        ax.axis("off")
    fig.legend(legend_handles(), CLASSES, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.suptitle("Inference on photos taken at ITB Ganesha campus", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(ASSETS / "itb_predictions.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    labels = {s: read_labels(d) for s, d in SPLITS.items()}
    fig_architecture()
    fig_pipeline()
    fig_dataset_stats(labels)
    fig_dataset_samples(labels)

    model = YOLO(str(WEIGHTS))
    res = evaluate(model)
    for split, r in res.items():
        p, rc, m50, m = r["all"]
        print(f"{split:5s} P={p:.3f} R={rc:.3f} mAP50={m50:.3f} mAP50-95={m:.3f}")
    fig_per_class(res)
    fig_gt_vs_pred(model, labels)
    fig_itb(model)
    print("Figures written to", ASSETS)


if __name__ == "__main__":
    main()
