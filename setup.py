# setup.py
from setuptools import setup, find_packages

setup(
    name="cloud-classifier",
    version="2.0.0",
    description="Cumulonimbus cloud detection via MobileNetV2 transfer learning",
    author="Cloud Classification Project",
    python_requires=">=3.9",
    packages=find_packages(include=["cli", "cli.*"]),
    py_modules=["config", "utils", "errors", "app"],
    install_requires=[
        "tensorflow>=2.15,<2.20",
        "opencv-python-headless>=4.8",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "click>=8.1",
        "rich>=13.0",
        "flask>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "cloud-classifier=cli.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
