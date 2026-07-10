"""Program 12 — small-object + satellite multi-label package."""
from .cifar import run_cifar
from .satellite import run_satellite
from .gradcam import gradcam

__all__ = ["run_cifar", "run_satellite", "gradcam"]
