import torch
import torchvision.models as models
from edgebridge.converters import TorchConverter
from edgebridge.core import Benchmark

def test_onnx_benchmark_tmp(tmp_path):
    model = models.resnet18(pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_file = tmp_path / "test_resnet.onnx"
    TorchConverter.to_onnx(model, dummy_input, export_path=str(onnx_file))
    bench = Benchmark(str(onnx_file), backend="onnx", device="cpu")
    res = bench.run(runs=3)
    assert "avg_latency_ms" in res
    assert isinstance(res["timings_ms"], list)
