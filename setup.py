from setuptools import setup, find_packages

setup(
    name="edgebridge",
    version="0.3.0",
    author="Aarav Mehta",
    author_email="",
    description="Bridge ML models to lightweight edge-ready formats with quantization, pruning, distillation, and benchmarking",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AaravMehta-07/edgebridge",
    packages=find_packages(),
    install_requires=[
        "onnxruntime",
        "torch",
        "torchvision",
        "numpy",
    ],
    extras_require={
        "tflite": ["tensorflow", "onnx", "onnx-tf"],
        "gui": ["pyqt6", "pyqtgraph"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
