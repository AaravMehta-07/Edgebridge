"""
Converters:
- TorchConverter.to_onnx
- onnx_to_tflite (requires onnx, onnx-tf, tensorflow)
- torch_to_tflite (helper pipeline)
"""

import os

def _ensure_torch():
    try:
        import torch
    except Exception as e:
        raise ImportError("torch is required for conversion functions") from e
    return torch

def _ensure_onnx_tf():
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except Exception as e:
        raise ImportError("onnx, onnx-tf, and tensorflow are required for ONNX->TFLite conversion") from e
    return onnx, prepare, tf

class TorchConverter:
    @staticmethod
    def to_onnx(model, sample_input, export_path="model.onnx", opset_version=12):
        torch = _ensure_torch()
        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                export_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        print(f"✅ Saved ONNX model at {export_path}")
        return export_path

def onnx_to_tflite(onnx_model_path, out_path="model.tflite", tmp_tf_dir="tmp_tf"):
    onnx, prepare, tf = _ensure_onnx_tf()

    # load onnx and convert to TF representation
    model = onnx.load(onnx_model_path)
    tf_rep = prepare(model)
    # export temp saved model
    if os.path.exists(tmp_tf_dir):
        # remove existing tmp dir to avoid conflicts
        import shutil
        shutil.rmtree(tmp_tf_dir)
    tf_rep.export_graph(tmp_tf_dir)

    # convert saved model -> tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_tf_dir)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Saved TFLite model at {out_path}")
    return out_path

def torch_to_tflite(torch_model, sample_input, out_path="model.tflite"):
    # Torch -> ONNX -> TFLite helper
    from tempfile import NamedTemporaryFile
    import torch
    with NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_name = tmp.name
    TorchConverter.to_onnx(torch_model, sample_input, export_path=tmp_name)
    return onnx_to_tflite(tmp_name, out_path=out_path)
