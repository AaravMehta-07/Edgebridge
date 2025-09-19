"""
Simple example: export a torchvision ResNet18 to ONNX, then benchmark locally.
"""

import torch
import torchvision.models as models
from edgebridge.converters import TorchConverter
from edgebridge.core import Benchmark

if __name__ == "__main__":
    model = models.resnet18(pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = TorchConverter.to_onnx(model, dummy, export_path="resnet18.onnx")
    bench = Benchmark(onnx_path, backend="onnx", device="cpu")
    results = bench.run(runs=10)
    import json
    print(json.dumps(results, indent=2))
