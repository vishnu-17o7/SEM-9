"""CV Lab router — experiment catalog and detail pages."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from hub.templates import render
from hub.supermemory import memorize, recall

router = APIRouter(prefix="/cv", tags=["cv"])

EXPERIMENTS = [
    {
        "id": "experiment_1_rectangle_transformations",
        "num": "1",
        "title": "2D Geometric Transformations on Rectangles",
        "type": "cv",
        "run_script": "run_1.py",
        "output_dir": "experiment_1_rectangle_transformations",
        "description": "Keyboard-controlled rectangle movement (translation) and mouse-controlled transformations including translation, scaling, rotation, and affine warp. Demonstrates basic OpenCV drawing and interactive event handling.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_1.py" run',
        "has_package": False,
        "extra_args": [],
        "ui_controls": "Keyboard arrows for translation mode. Mouse drag for affine transformations. 'q' or ESC to quit.",
        "models": [
            "Keyboard translation (WASD, delta clipping)",
            "Mouse translation (drag, cursor tracking)",
            "Scaling (uniform, 0.5x–3.0x, click/keys)",
            "Rotation (30° steps, 2D rotation matrix)",
            "Affine shear (off-diagonal matrix elements)",
        ],
        "datasets": [],
        "comparisons": "Original rectangle vs Translated vs Scaled vs Rotated vs Affine-warped — all interactive real-time in two live OpenCV windows.",
    },
    {
        "id": "experiment_2_image_restoration",
        "num": "2",
        "title": "Vintage Photo Restoration Pipeline",
        "type": "cv",
        "run_script": "run_2.py",
        "output_dir": "experiment_2_image_restoration",
        "description": "Three-stage restoration pipeline on a vintage 1910s New York photograph: (A) blending with linear and Poisson techniques, (B) morphing via cross-dissolve, (C) edge enhancement with unsharp mask and Sobel/Scharr. Damage simulates authentic vintage photo degradation: yellowing, scratches, fading, and dust.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_2.py" run',
        "has_package": False,
        "extra_args": [],
        "ui_controls": "Interactive sliders for blend parameters. 'n'/'p' for next/previous sample. 's' to save. 'q' or ESC to quit.",
        "models": [
            "Linear blending (cv2.addWeighted, alpha slider)",
            "Poisson blending (cv2.seamlessClone, NORMAL_CLONE)",
            "Cross-dissolve morphing (8-step linear interpolation)",
            "Unsharp mask (Gaussian blur + subtract, strength=1.5)",
            "Sobel / Scharr edge detection (gradient magnitude)",
        ],
        "datasets": [
            "Vintage Flatiron Building photograph (Library of Congress, 1903, real photo, public domain)",
        ],
        "comparisons": "Original vintage photo vs Damaged (sepia yellowing, scratches, dust, vignette fading) vs Linear blend vs Poisson blend vs 8-step morph sequence vs Scharr edge map vs Final edge-enhanced restored image.",
    },
    {
        "id": "experiment_3_feature_detectors",
        "num": "3",
        "title": "SIFT / AKAZE / ORB Landmark Keypoint Detection",
        "type": "cv",
        "run_script": "run_3.py",
        "output_dir": "experiment_3_feature_detectors",
        "description": "4-panel display per landmark: original image vs SIFT, AKAZE, and ORB keypoint locations on 4 famous landmarks (Statue of Liberty, Eiffel Tower, Colosseum, Taj Mahal). Coloured circles at detected keypoints with no connecting lines.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_3.py" run',
        "has_package": False,
        "extra_args": [],
        "ui_controls": "'n'/'p' to cycle through 4 landmarks. 's' to save current panel. 'q' or ESC to quit.",
        "models": [
            "SIFT (Difference-of-Gaussians, scale-space extrema, nfeatures=5000, Lowe 2004)",
            "AKAZE (Accelerated KAZE, non-linear diffusion, MLDB descriptor, Alcantarilla 2013)",
            "ORB (FAST corners with orientation, nfeatures=120000, Rublee 2011)",
        ],
        "datasets": [
            "Statue of Liberty (direct photograph)",
            "Eiffel Tower, Colosseum, Taj Mahal (Picsum landmark photographs)",
        ],
        "comparisons": "2x2 grid per landmark: Original vs SIFT keypoints (cyan) vs AKAZE keypoints (green) vs ORB keypoints (orange). Keypoint count + latency per panel. Cycle through all 4 landmarks with n/p keys. No connecting lines — only landmark point positions.",
    },
    {
        "id": "experiment_4_rgb_depth",
        "num": "4",
        "title": "Multi-Object Detection + Segmentation (RGB + Depth)",
        "type": "cv",
        "run_script": "run_4.py",
        "output_dir": "experiment_4_rgb_depth",
        "description": "Combines YOLOv3-tiny object detection with MiDaS depth estimation. Detects objects in RGB images and estimates depth maps. Shows both overlays simultaneously with configurable confidence thresholds.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_4.py" run',
        "has_package": True,
        "extra_args": [],
        "ui_controls": "Sliders for detection confidence and depth percentile stretch. 'd' to toggle depth colormap. 'n' for next sample. 's' to save. 'q' or ESC to quit.",
        "models": [
            "YOLOv3-tiny (Redmon 2018, Darknet backbone, OpenCV DNN)",
            "MiDaS-small (Ranftl 2021, ONNX, monocular depth estimation)",
            "3D box projection (depth + detection to perspective wireframe)",
        ],
        "datasets": [
            "Picsum photos (5 diverse scenes: indoor, street, parking, kitchen, park)",
            "COCO 80-class labels (OpenCV DNN, NMS post-processing)",
        ],
        "comparisons": "RGB detection boxes (YOLOv3-tiny) vs Depth map (MiDaS-small, inferno/viridis/jet colormaps) vs 3D projected boxes — three-panel side-by-side with slider-controlled confidence and depth percentile.",
    },
    {
        "id": "experiment_5_ocr_captioning",
        "num": "5",
        "title": "OCR + Image Captioning",
        "type": "cv",
        "run_script": "run_5.py",
        "output_dir": "experiment_5_ocr_captioning",
        "description": "Performs OCR text extraction and generates image captions using pre-trained models. Displays detected text regions and generated captions overlaid on input images.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_5.py" run',
        "has_package": False,
        "extra_args": [],
        "ui_controls": "'o' to toggle OCR boxes. 'c' to toggle caption overlay. 'n'/'p' for next/previous. 's' to save. 'q' or ESC to quit.",
        "models": [
            "Tesseract OCR (Smith 2007, pytesseract, confidence >= 30)",
            "BLIP image captioning (Li 2022, Salesforce/blip-image-captioning-base)",
            "MSER region detector (fallback OCR, OpenCV)",
            "Template-based caption fallback (HSV brightness/saturation + edge density)",
        ],
        "datasets": [
            "Synthetic scenes: chalkboard menu, directional street sign, newspaper/book spread",
        ],
        "comparisons": "OCR extracted text vs BLIP generated caption vs template fallback caption — toggle overlays per image with engine labels and confidence scores.",
    },
    {
        "id": "experiment_6_yolo_maskrcnn",
        "num": "6",
        "title": "YOLOv3 + Mask R-CNN on Traffic Video",
        "type": "cv",
        "run_script": "run_6.py",
        "output_dir": "experiment_6_yolo_maskrcnn",
        "description": "Compares YOLOv3 (single-shot) and Mask R-CNN (two-stage) object detectors on traffic video. Shows bounding boxes, confidence scores, and instance segmentation masks side-by-side.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_6.py" run',
        "has_package": True,
        "extra_args": [],
        "ui_controls": "Sliders for YOLO and Mask R-CNN confidence thresholds. Space to pause/resume. 's' to save frame. 'r' to rewind. 'q' or ESC to quit.",
        "models": [
            "YOLOv3 (Redmon 2018, Darknet-53 backbone, OpenCV DNN, NMS)",
            "Mask R-CNN (He 2017, ResNet-50 FPN, torchvision, COCO-pretrained)",
        ],
        "datasets": [
            "Synthetic traffic video (60 frames, 10 FPS, COCO 80 classes)",
        ],
        "comparisons": "Side-by-side: YOLOv3 bounding boxes vs Mask R-CNN instance segmentation masks + boxes. Live per-frame detection counts, total counts CSV, and video export (out.mp4). Slider-controlled confidence thresholds.",
    },
    {
        "id": "experiment_7_defect_detection",
        "num": "7",
        "title": "Custom Object Detection for Industrial Defects",
        "type": "cv",
        "run_script": "run_7.py",
        "output_dir": "experiment_7_defect_detection",
        "description": "Trains a Faster R-CNN model for industrial defect detection. Generates synthetic defect datasets, trains with real-time loss visualization, and evaluates on validation images with ground truth vs prediction overlays.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_7.py" run --epochs 4 --n-synth 120',
        "has_package": True,
        "extra_args": [
            "python run_7.py --epochs 4 --n-synth 120"
        ],
        "ui_controls": "Training: real-time loss curves in matplotlib. Inference: 'n' for next batch. 'q' to quit.",
        "models": [
            "Faster R-CNN (Ren 2015, ResNet-50 FPN, torchvision, ImageNet-pretrained)",
        ],
        "datasets": [
            "Synthetic defect dataset (scratch / dent / hole on brushed metal, 120 train + 24 val)",
            "Optional MVTec AD subset (single-class per folder, train/val splits)",
        ],
        "comparisons": "Ground truth bounding boxes (green) vs Faster R-CNN predicted boxes (cyan) on validation images. Training loss curves (train/val over epochs). Per-class defect metrics.",
    },
    {
        "id": "experiment_8_flipbook_registration",
        "num": "8",
        "title": "Feature Matching + Image Registration (Flipbook)",
        "type": "cv",
        "run_script": "run_8.py",
        "output_dir": "experiment_8_flipbook_registration",
        "description": "Creates a flipbook animation by registering sequential images. Uses feature matching (ORB) to align frames and generates a stabilized video playback.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_8.py" run',
        "has_package": False,
        "extra_args": [],
        "ui_controls": "Space to pause/resume playback. Left/Right arrows to step. +/- to adjust FPS. 'q' or ESC to quit.",
        "models": [
            "ORB (Rublee 2011, n=1500 features, FAST corners + BRIEF descriptors)",
            "BFMatcher (Hamming distance, cross-check)",
            "RANSAC homography estimation (3.0 px threshold)",
            "Perspective warp (cv2.warpPerspective, frame-to-frame stabilization)",
        ],
        "datasets": [
            "Synthetic flipbook sequence (12 frames: landscape with moving car, animated sun)",
            "Optional user-supplied image sequence (>=3 frames in sequence/)",
        ],
        "comparisons": "Unregistered input frames vs ORB-matched + Homography-warped + Registered/aligned frames. Per-frame-pair inlier counts. Output: contact sheet, GIF animation, MP4 video.",
    },
    {
        "id": "experiment_9_vggface2",
        "num": "9",
        "title": "Face Recognition with VGGFace2",
        "type": "cv",
        "run_script": "run_9.py",
        "output_dir": "experiment_9_vggface2",
        "description": "Face detection, embedding, and recognition pipeline using VGGFace2. Builds a gallery of known faces, then performs detect/identify/verify modes on probe images with cosine similarity scoring.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_9.py" run --mode identify --threshold 0.6',
        "has_package": True,
        "extra_args": [
            "python run_9.py --mode identify --threshold 0.6",
        ],
        "ui_controls": "Slider for cosine threshold. 'm' to toggle mode (detect/identify/verify). 'n'/'p' for next/previous probe. 's' to save. 'q' or ESC to quit.",
        "models": [
            "MTCNN face detection (Zhang 2016, facenet-pytorch, multi-task cascaded CNNs)",
            "VGGFace2 ResNet-50 SE (keras-vggface, 2048-D pooled embedding)",
            "FaceNet InceptionResNet-V1 (Schroff 2015, facenet-pytorch, VGGFace2-pretrained, 512-D)",
            "Cosine similarity scoring (L2-normalised dot product, threshold sweep)",
        ],
        "datasets": [
            "Local gallery folders (user-provided, one folder per identity)",
        ],
        "comparisons": "Detect mode (all faces) vs Identify mode (best match + cosine score) vs Verify mode (ACCEPT/REJECT by threshold). Live threshold slider (0-1.0) and three-mode toggle.",
    },
    {
        "id": "experiment_10_facenet",
        "num": "10",
        "title": "FaceNet Embeddings for Face Classification",
        "type": "cv",
        "run_script": "run_10.py",
        "output_dir": "experiment_10_facenet",
        "description": "Uses FaceNet embeddings to train a classifier (SVM or KNN) for face recognition. Shows training curves, confusion matrix, and sample predictions.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_10.py" run --classifier svm --test-size 0.25',
        "has_package": True,
        "extra_args": [
            "python run_10.py --classifier svm --test-size 0.25",
        ],
        "ui_controls": "Training: real-time accuracy curves in matplotlib. After training: confusion matrix + sample predictions. 's' to save. 'q' to quit.",
        "models": [
            "FaceNet InceptionResNet-V1 (Schroff 2015, VGGFace2-pretrained, 512-D L2 embedding)",
            "SVM (RBF kernel, scikit-learn)",
            "KNN classifier (scikit-learn)",
        ],
        "datasets": [
            "LFW subset — Labeled Faces in the Wild (sklearn fetch_lfw_people, min 12 faces/person)",
            "Local face folders (one folder per identity, .jpg/.png)",
        ],
        "comparisons": "FaceNet 512-D embeddings to scikit-learn classifier (SVM vs KNN) to train/test split to Confusion matrix + accuracy + per-class precision/recall/F1. Live accuracy curve.",
    },
    {
        "id": "experiment_11_fashion_mnist",
        "num": "11",
        "title": "CNN Classifier on Fashion-MNIST",
        "type": "cv",
        "run_script": "run_11.py",
        "output_dir": "experiment_11_fashion_mnist",
        "description": "Trains a convolutional neural network on the Fashion-MNIST dataset. Displays real-time training curves and a sample predictions grid after training.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_11.py" run --epochs 5',
        "has_package": False,
        "extra_args": [],
        "ui_controls": "Training: real-time loss/accuracy curves in matplotlib. After training: 3x6 sample predictions grid. 's' to save. 'q' to quit.",
        "models": [
            "Custom CNN: 2x Conv2D(32)-BatchNorm-ReLU + MaxPool(2) + Dropout(0.25) + 2x Conv2D(64)-BatchNorm-ReLU + MaxPool(2) + Dropout(0.25) + Flatten + FC(128) + Dropout(0.5) + 10-way softmax",
        ],
        "datasets": [
            "Fashion-MNIST (Xiao 2017, 60k grayscale 28x28, 10 clothing classes)",
        ],
        "comparisons": "Training curves (loss/accuracy over epochs) to Confusion matrix (10x10 with per-class labels) to Sample predictions grid (3x6 with true vs predicted labels). Final test accuracy and classification report.",
    },
    {
        "id": "experiment_12_smallobj_satellite",
        "num": "12",
        "title": "Small-object + Satellite Multi-label Classification",
        "type": "cv",
        "run_script": "run_12.py",
        "output_dir": "experiment_12_smallobj_satellite",
        "description": "Two-task experiment: (A) CIFAR-10 small object classification and (B) satellite image multi-label classification. Shows Grad-CAM visualizations for interpretability.",
        "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\run_12.py" run --epochs 3 --quick',
        "has_package": True,
        "extra_args": [
            "python run_12.py --epochs 3 --quick",
        ],
        "ui_controls": "Training: one matplotlib window per sub-task with loss curves. After training: Grad-CAM grid. 's' to save. 'q' to quit.",
        "models": [
            "ResNet-18 (He 2016, ImageNet-pretrained, torchvision, fine-tuned for 10-class and multi-label)",
            "Grad-CAM (Selvaraju 2017, layer4 backward gradient heatmaps)",
        ],
        "datasets": [
            "CIFAR-10 (Krizhevsky 2009, 60k 32x32 colour images, 10 classes: airplane to truck)",
            "EuroSAT-style synthetic multi-label dataset (5 land-cover classes: water, forest, urban, agricultural, barren)",
        ],
        "comparisons": "Task A: CIFAR-10 confusion matrix + per-class accuracy. Task B: EuroSAT multi-label micro-F1 + per-label F1. Both tasks: Grad-CAM heatmaps (layer4, ResNet-18) superimposed on input images for interpretability.",
    },
]

CV_EXPERIMENT_MAP = {e["id"]: e for e in EXPERIMENTS}


@router.get("", response_class=HTMLResponse)
async def cv_index(request: Request):
    html = render("cv/index.html", request=request, experiments=EXPERIMENTS, active="cv")
    return HTMLResponse(html)


@router.get("/{experiment_id}", response_class=HTMLResponse)
async def cv_experiment(request: Request, experiment_id: str):
    exp = CV_EXPERIMENT_MAP.get(experiment_id)
    if not exp:
        return HTMLResponse("Experiment not found", status_code=404)

    memorize(
        content=f"Viewed CV experiment '{exp['title']}' (id: {experiment_id})",
        container_tag="sem9-hub",
        metadata={
            "experiment_id": experiment_id,
            "experiment_title": exp["title"],
            "type": "page_view",
            "lab": "cv",
        },
    )

    past_runs = recall(query=f"CV {exp['title']} script_run", limit=5)

    html = render(
        "cv/experiment.html",
        request=request,
        experiment=exp,
        active="cv",
        memories=past_runs,
    )
    return HTMLResponse(html)
