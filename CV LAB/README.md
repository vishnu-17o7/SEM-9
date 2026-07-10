# CV Lab

A complete set of Computer Vision lab experiments. Every program has a
**live `cv2` UI** (and a live `matplotlib` training window for the deep
learning ones), so the same way Program 1 lets you steer a rectangle
with the keyboard, every program in this lab gives you a live view you
can interact with.

## Quick Start

Each experiment has a unified `run_N.py` entry point:

```bash
# From the SEM 9 root
.venv\Scripts\python.exe "CV LAB\run_3.py" all

# From inside CV LAB/
python run_3.py run          # Launch interactive OpenCV GUI
python run_3.py steps        # List available commands
```

Examples with arguments:

```bash
python run_7.py run --epochs 8 --n-synth 120    # Custom defect detection
python run_11.py run --epochs 10 --quick          # Fashion-MNIST quick training
python run_9.py run --mode verify --threshold 0.5 # Face recognition
```

## Layout

```
CV LAB/
  _common.py                              # shared helpers + live UI
  experiment_1_rectangle_transformations.py
  experiment_2_image_restoration.py
  experiment_3_feature_detectors.py
  experiment_4_rgb_depth.py               (alias → program_4_rgb_depth/)
  experiment_5_ocr_captioning.py
  experiment_6_yolo_maskrcnn.py           (alias → program_6_yolo_maskrcnn/)
  experiment_7_defect_detection.py        (alias → program_7_defect_detection/)
  experiment_8_flipbook_registration.py
  experiment_9_vggface2.py                (alias → program_9_vggface2/)
  experiment_10_facenet.py                (alias → program_10_facenet/)
  experiment_11_fashion_mnist.py
  experiment_12_smallobj_satellite.py     (alias → program_12_smallobj_satellite/)

  program_4_rgb_depth/    # multi-file
  program_6_yolo_maskrcnn/
  program_7_defect_detection/
  program_9_vggface2/
  program_10_facenet/
  program_12_smallobj_satellite/

  assets/                                 # place models / datasets / images here
    cache/  models/  datasets/  samples/  sequence/  landmarks/  gallery/  probes/  videos/
  outputs/                                # per-experiment output folders
```

## Quick start

Open `dashboard.html` for a polished index of all experiments, commands,
asset folders, and output locations.

```bash
# from the SEM 9/ root (the .venv at the root already has the base deps)
.venv\Scripts\python.exe "CV LAB/experiment_2_image_restoration.py"
.venv\Scripts\python.exe "CV LAB/experiment_4_rgb_depth.py"
.venv\Scripts\python.exe "CV LAB/experiment_11_fashion_mnist.py" --epochs 5
```

## Live UI conventions

Every program uses the same dark-themed info panel (matching Program 1)
and the same key bindings where they make sense.

| Key / Slider        | Action                                              |
|---------------------|-----------------------------------------------------|
| `q` / `Esc`         | Quit                                                |
| `s`                 | Save current frame as PNG                           |
| `n` / `p`           | Next / previous sample                              |
| `space`             | Pause / resume (video + flipbook)                   |
| `←` / `→`           | Step back / forward                                 |
| `+` / `-`           | Bump slider step / playback FPS                     |
| `m`                 | Toggle mode overlay                                 |
| `h`                 | Toggle help                                         |
| `r`                 | Reset                                               |
| Trackbar sliders    | Live parameter tuning (conf, FPS, depth, etc.)      |
| Matplotlib window   | Live training curves (deep-learning programs)       |

## Mapping — program → file

| #  | Topic                                                  | Entry point                                          | Runner           |
|----|--------------------------------------------------------|------------------------------------------------------|------------------|
| 1  | 2D rectangle transformations (translation, scale, etc) | `experiment_1_rectangle_transformations.py`          | `run_1.py`       |
| 2  | Image restoration (blend / morph / edge enhance)       | `experiment_2_image_restoration.py`                  | `run_2.py`       |
| 3  | SIFT / SURF / ORB feature detectors                    | `experiment_3_feature_detectors.py`                  | `run_3.py`       |
| 4  | Multi-object detection + depth (RGB + MiDaS)           | `program_4_rgb_depth/main.py`                        | `run_4.py`       |
| 5  | OCR (Tesseract) + image captioning (BLIP / template)   | `experiment_5_ocr_captioning.py`                     | `run_5.py`       |
| 6  | YOLOv3 + Mask R-CNN on traffic video                   | `program_6_yolo_maskrcnn/main.py`                    | `run_6.py`       |
| 7  | Custom defect detection (Faster R-CNN)                 | `program_7_defect_detection/main.py`                 | `run_7.py`       |
| 8  | Feature matching + flipbook registration               | `experiment_8_flipbook_registration.py`              | `run_8.py`       |
| 9  | Face recognition (VGGFace2 / FaceNet)                  | `program_9_vggface2/main.py`                         | `run_9.py`       |
| 10 | FaceNet classification on face embeddings              | `program_10_facenet/main.py`                         | `run_10.py`      |
| 11 | Fashion-MNIST CNN (grayscale clothing)                 | `experiment_11_fashion_mnist.py`                     | `run_11.py`      |
| 12 | CIFAR-10 + EuroSAT multi-label + Grad-CAM              | `program_12_smallobj_satellite/main.py`              | `run_12.py`      |

## Per-program UI specifics

- **Program 2** — slider `alpha x100` controls the linear-blend view in real time; `n`/`p` cycle through 6 visualisation variants (damaged / linear / Poisson / morph sheet / edges / final).
- **Program 3** — three detectors stacked per landmark pair; `m` toggles the metrics overlay; `n`/`p` cycle pairs.
- **Program 4** — three panels side by side (RGB | depth | 3D boxes); sliders for `conf` and `depth_pct`; `d` cycles depth colormaps; `n` cycles samples.
- **Program 5** — `o` toggles OCR boxes, `c` toggles the caption strip; `n`/`p` cycle images.
- **Program 6** — playable video with progress bar and live class counts; `space` pauses, `r` rewinds, sliders for YOLO and Mask R-CNN confidence.
- **Program 7** — live matplotlib loss curves during training, then a 2x3 val grid (GT in green, predictions in cyan) for live review.
- **Program 8** — flipbook player; `space` pause, `←/→` step, `+/-` change FPS, `f` flip horizontally, `r` reset.
- **Program 9** — `m` switches between `detect` / `identify` / `verify` modes; slider `threshold x100` for verify; `n`/`p` cycle probes.
- **Program 10** — live matplotlib training curves (train vs val accuracy) + a confusion-matrix viewer at the end.
- **Program 11** — live matplotlib loss + accuracy curves during training, then a sample-prediction grid (`g`) and confusion matrix (`c`).
- **Program 12** — one matplotlib window per sub-task; Grad-CAM grid viewer at the end.

## Asset placement (when programs need real weights)

Most programs work out of the box (synthetic data / auto-fetched
samples / lightweight defaults). To use the real heavyweight
models, drop them into the paths below — the program prints a
download hint if anything is missing.

- YOLOv3: `assets/models/{yolov3.weights, yolov3.cfg, coco.names}`
- YOLOv3-tiny: `assets/models/{yolov3-tiny.weights, yolov3-tiny.cfg, coco.names}`
- Mask R-CNN: auto-downloaded by torchvision on first run
- MiDaS-small: `assets/models/midas_v21_small_256.onnx`
- VGGFace2 weights: `assets/models/vggface2_resnet50.h5` (or rely on the
  facenet-pytorch fallback that's bundled with VGGFace2-pretrained weights)
- Custom gallery: `assets/gallery/<name>/*.jpg`
- Custom probes: `assets/probes/*.jpg`
- MVTec-style defects: `assets/datasets/mvtec_subset/<class>/<split>/*`

See each `assets/<subfolder>/README.md` for the exact layout.
