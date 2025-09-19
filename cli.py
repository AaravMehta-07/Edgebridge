"""
Simple CLI entry point for edgebridge.
Supports quick convert/quantize/benchmark flows.
"""

import argparse
import sys
from pathlib import Path
from .converters import TorchConverter, onnx_to_tflite
from .optimizer import Optimizer
from .core import Benchmark

def main(argv=None):
    parser = argparse.ArgumentParser(prog="edgebridge", description="EdgeBridge CLI")
    sub = parser.add_subparsers(dest="cmd")

    # benchmark
    b = sub.add_parser("benchmark", help="Run benchmark")
    b.add_argument("--model", required=True)
    b.add_argument("--backend", default="onnx", choices=["onnx","tflite","torchscript"])
    b.add_argument("--runs", type=int, default=20)

    # to-onnx
    t = sub.add_parser("to-onnx", help="Export torch model to ONNX (scripted torch model recommended)")
    t.add_argument("--model", required=True)
    t.add_argument("--out", default="model.onnx")

    # quantize
    q = sub.add_parser("quantize", help="Quantize PyTorch model (dynamic/static)")
    q.add_argument("--model", required=True)
    q.add_argument("--mode", default="dynamic", choices=["dynamic","static"])

    args = parser.parse_args(argv)

    if args.cmd == "benchmark":
        rpt = Benchmark(args.model, backend=args.backend).run(runs=args.runs)
        import json
        print(json.dumps(rpt, indent=2))
        return 0

    if args.cmd == "to-onnx":
        import torch
        model = torch.jit.load(args.model) if Path(args.model).suffix in [".pt",".pth"] else None
        if model is None:
            print("Please provide a scripted/traced .pt torch model for reliable export.")
            return 1
        TorchConverter.to_onnx(model, torch.randn(1,3,224,224), export_path=args.out)
        return 0

    if args.cmd == "quantize":
        import torch
        model = torch.jit.load(args.model) if Path(args.model).suffix in [".pt",".pth"] else None
        if model is None:
            print("Please provide a scripted/traced .pt torch model for quantization.")
            return 1
        opt = Optimizer(model, None)
        qm = opt.quantize(mode=args.mode)
        out = Path(args.model).with_suffix(f".quant.{args.mode}.pt")
        try:
            import torch
            torch.jit.save(torch.jit.script(qm), str(out))
            print(f"Saved quantized scripted model at {out}")
        except Exception as e:
            print(f"Could not save scripted quantized model: {e}")
        return 0

    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
