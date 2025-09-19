"""
Runners for different model formats.
ONNX -> onnxruntime
TFLite -> tensorflow.lite.Interpreter
TorchScript -> torch.jit
"""

import numpy as np

# Lazy-import heavy libs inside classes to avoid import errors for lightweight usage.

class OnnxRunner:
    def __init__(self, model_path, device="cpu"):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError("onnxruntime is required for OnnxRunner") from e

        providers = ["CPUExecutionProvider"]
        if device and device.lower().startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, inp):
        # Ensure numpy contiguous float32
        arr = np.array(inp, dtype=np.float32)
        return self.session.run(None, {self.input_name: arr})[0]


class TFLiteRunner:
    def __init__(self, model_path, device="cpu"):
        try:
            import tensorflow as tf
        except Exception as e:
            raise ImportError("tensorflow is required for TFLiteRunner") from e

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, inp):
        import numpy as _np
        arr = _np.array(inp, dtype=_np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], arr)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


class TorchScriptRunner:
    def __init__(self, model_path, device="cpu"):
        try:
            import torch
        except Exception as e:
            raise ImportError("torch is required for TorchScriptRunner") from e
        dev = device if device else "cpu"
        self.model = torch.jit.load(model_path, map_location=dev)
        self.model.eval()

    def predict(self, inp):
        import torch as _torch
        arr = _torch.tensor(inp).to(next(self.model.parameters()).device) if hasattr(self.model, 'parameters') else _torch.tensor(inp)
        with _torch.no_grad():
            out = self.model(arr)
        return out.cpu().numpy()
