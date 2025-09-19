from setuptools import setup, find_packages

setup(
    name="edgebridge",
    version="0.1.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="Edgebridge is a versatile Python library offering AI, ML, and edge computing utilities, streamlining tasks like model handling, data processing, deployment, and automation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/EdgeBridge",
    packages=find_packages(include=["edgebridge", "edgebridge.*"]) + [
        "utils", "core", "optimizer", "converters", "cli", "runners"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        # add any other dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

