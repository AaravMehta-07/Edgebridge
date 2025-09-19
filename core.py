"""
Benchmark core.

Now returns per-iteration timings in addition to aggregate metrics.
"""

import time
import numpy as np
from .runners import OnnxRunner, TFLiteRunner, TorchScriptRunner
from .utils import get_logger

logger = get_logger("edgebridge.core")

class Benchmark:
    """
    Unified Benchmark Class for ONNX, TFLite, TorchScript models.
    """

    def __init__(self, model_path, backend="onnx", device="cpu", warmup=5):
        self.model_path = model_path
        self.backend = backend.lower()
        self.device = device
        self.warmup = warmup

        if self.backend == "onnx":
            self.runner = OnnxRunner(model_path, device)
        elif self.backend == "tflite":
            self.runner = TFLiteRunner(model_path, device)
        elif self.backend == "torchscript":
            self.runner = TorchScriptRunner(model_path, device)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def run(self, input_shape=(1, 3, 224, 224), runs=50):
        """
        Run benchmark on given model.

        Returns:
            dict with keys:
                - backend, device
                - avg_latency_ms, median_ms, p95_ms, throughput_fps
                - timings_ms: list of per-iteration timings
        """
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

        # Warmup
        logger.info("Warming up...")
        for _ in range(self.warmup):
            self.runner.predict(dummy_input)

        times = []
        logger.info(f"Running {runs} timed iterations...")
        for i in range(runs):
            t0 = time.perf_counter()
            self.runner.predict(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        times_np = np.array(times)
        avg_latency = float(np.mean(times_np))
        median = float(np.median(times_np))
        p95 = float(np.percentile(times_np, 95))
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        report = {
            "backend": self.backend,
            "device": self.device,
            "avg_latency_ms": round(avg_latency, 4),
            "median_ms": round(median, 4),
            "p95_ms": round(p95, 4),
            "throughput_fps": round(throughput, 4),
            "timings_ms": [round(float(x), 4) for x in times],
        }
        return report
