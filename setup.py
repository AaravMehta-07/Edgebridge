from setuptools import setup, find_packages

setup(
    name="edgebridge",
    version="0.1.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="Edgebridge is a versatile Python library offering AI, ML, and edge computing utilities.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/EdgeBridge",
    packages=find_packages(),  # This will now only include the edgebridge/ folder
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        # add other dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
