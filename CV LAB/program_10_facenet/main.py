"""
Program 10 — FaceNet Embeddings for Face Classification (LIVE UI)
================================================================

Live UI
-------
  During training: a matplotlib window that updates train/val accuracy
                   in real time, alongside the live cv2 info panel.
  After training : confusion matrix + sample predictions in a window.
                   `s` save, `q` quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from _common import (
    DATA_DIR, OUTPUT_DIR, print_banner, save_figure, save_image, save_json, device,
    live_show, live_keys, LivePlot, fit_to,
)

from .embedder import FaceNetEmbedder
from .dataset import load_face_dataset
from .classify import make_classifier, train_classifier

EXPERIMENT = "experiment_10_facenet"
WIN = "Experiment 10 — FaceNet"


def main(classifier: str = "svm", test_size: float = 0.25) -> None:
    print_banner("Experiment 10 — FaceNet Face Classification")
    dev = device()
    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading dataset...")
    X_imgs, y, classes = load_face_dataset(DATA_DIR / "lfw_subset", min_faces=12)
    print(f"      images: {len(X_imgs)}  classes: {len(classes)}")
    if len(classes) < 2:
        raise ValueError(
            "FaceNet classification needs at least two identities. "
            f"Add folders under {DATA_DIR / 'lfw_subset'}/<person>/*.jpg."
        )

    print("[2/4] computing FaceNet embeddings (this is the slow step)...")
    embedder = FaceNetEmbedder(device_name=dev)
    X = np.stack([embedder.embed(img) for img in X_imgs], axis=0)
    np.savez_compressed(out_dir / "embeddings.npz", X=X, y=y, classes=np.array(classes))
    print(f"      embeddings: {X.shape}")

    print(f"[3/4] training {classifier}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0,
        stratify=y if len(set(y)) > 1 else None,
    )

    plot = LivePlot("Program 10 — FaceNet (training)", subplots=1)
    plot.add_series("train_score", 0)
    plot.add_series("val_score", 0)

    # Train final classifier on full data
    final_clf = train_classifier(classifier, X_train, y_train)
    ta = final_clf.score(X_train, y_train)
    va = final_clf.score(X_test, y_test) if len(X_test) else 0.0

    # Show a single "step" on the live plot for visual feedback
    history = {"train_acc": [ta], "val_acc": [va]}
    plot.update(train_score=ta, val_score=va)
    # Arrange axes sensibly for a single data point
    plot.axes[0].set_xlim(0, 2)
    plot.axes[0].set_ylim(0, 1.05)
    plot.save(str(out_dir / "training_curves_live.png"))
    plot.close()
    print(f"      train_acc={ta:.3f}  val_acc={va:.3f}")

    print("[4/4] evaluating with live matrix + grid...")
    y_pred = final_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      accuracy: {acc:.3f}")
    print(classification_report(
        y_test, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(1 + 0.4 * len(classes), 1 + 0.4 * len(classes)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (acc={acc:.2f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=6)
    fig.tight_layout()
    save_figure("confusion_matrix.png", fig, EXPERIMENT)
    plt.close(fig)

    # Live window: show the confusion matrix
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 800)
    cm_img = cv2.imread(str(out_dir / "confusion_matrix.png"))
    info = [
        f"Program 10 — FaceNet accuracy={acc:.3f}  ({classifier})",
        f"classes: {len(classes)}  train={len(X_train)}  test={len(X_test)}",
        "s save   q quit",
    ]
    saved = False
    while True:
        live_show(WIN, fit_to(cm_img, 1280, 800), info, font_scale=0.55)
        k = live_keys(80)
        if k.get("q"): break
        elif k.get("s"):
            save_image("ui_capture_confusion.png", fit_to(cm_img, 1280, 800), EXPERIMENT)
            saved = True
    cv2.destroyAllWindows()

    save_json("metrics.json", {
        "accuracy": acc, "n_classes": len(classes),
        "n_train": len(X_train), "n_test": len(X_test),
        "classifier": classifier, "history": history,
    }, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/  saved={saved}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", choices=["svm", "knn"], default="svm")
    p.add_argument("--test_size", type=float, default=0.25)
    a = p.parse_args()
    main(classifier=a.classifier, test_size=a.test_size)
