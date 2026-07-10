"""Program 7 — Industrial defect detection package."""
from .dataset import build_or_load_dataset
from .model import build_faster_rcnn
from .train import train
from .infer import evaluate_and_visualise

__all__ = ["build_or_load_dataset", "build_faster_rcnn", "train", "evaluate_and_visualise"]
