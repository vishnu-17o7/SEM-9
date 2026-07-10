# Place model weight files here.

Each experiment expects its weights at a known path; the script will print
an exact `FileNotFoundError` hint if anything is missing.

Suggested filenames (you can rename, just update the script's `MODEL_DIR`
constant if you do):

- `yolov3.weights`, `yolov3.cfg`, `coco.names`            — Program 6
- `mask_rcnn/maskrcnn_resnet50_fpn.onnx`                  — Program 6
- `midas_v21_small_256.onnx`                              — Program 4
- `yolov5n.onnx` or `yolov8n.onnx`                        — Program 4
- `vggface2_resnet50.h5`                                  — Program 9
- `facenet_20180402_114759.pb`                            — Program 10
- `craft.pth`                                             — Program 5 (optional)
- `yolov8n.pt`                                            — Program 7 (auto-fetched by Ultralytics)
