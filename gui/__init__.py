# edgebridge/__init__.py

"""
EdgeBridge: Lightweight AI Model Conversion, Quantization, Pruning & Deployment Toolkit
Author: Aarav Mehta
GitHub: https://github.com/AaravMehta-07
"""

__version__ = "0.1.0"

# Import all core functions so users can access them directly from edgebridge
from .core import (
    convert_tf_to_tflite,
    convert_torch_to_onnx,
    convert_onnx_to_tf,
    run_inference_tf,
    run_inference_onnx,
    run_inference_tflite,
    benchmark_inference,
    quantize_model,
    prune_model,
    distill_model,
)

# Optional: import GUI/CLI functions if you have them
try:
    from .gui import launch_gui
except ImportError:
    pass  # GUI dependencies not installed

try:
    from .cli import main as cli_main
except ImportError:
    pass  # CLI dependencies not installed
