# 📷 Essential Image Processing Tools

A miscelaneous collection of Python libraries for basic image processing.

<!-- <div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python 3.8+ Required">
  <img src="https://img.shields.io/github/license/yourusername/project-name?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/last-commit/yourusername/project-name?style=for-the-badge" alt="Last Commit">
  </div> -->

---

## ✨ Features

This repository is structured to provide a set of tools and its corerresponding examples:

- **`pytools/`**: functions for image registration, slicing, anlysis, etc.
- **`gui_tools/`**: GUI tools used for image acquisition, processing, and analysis.
- **`exmples/`**: directory shows practical usage for every library.

---

## 🚀 Getting Started

### Prerequisites

- You will need **Python ≥ 3.8** installed on your system.

**NOTE**: Most of these scripts where designed and tested with `Python 3.13`, but they should work with 3.8 or newer.

---

## 📖 Library description

### 1. `improc_tools`

This library handles different libraries for image processing.

| Script             | Description                                                                                                                      | Examples                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `DFT_registration` | Fast DFT-based image rigid registration with Numpy and Pytorch implementations.                                                  | DFT_registration_example.ipynb |
| `ImageSlicer.py`   | Extracts a rectangular ROI (can be rotated) slice from an image and straightens it. Useful for analyzing small ROIs in an image. | ImageSlicer_example.ipynb      |
| `PatternMaker.py`  | Creates image with different patterns.                                                                                           | PatternMaker_example.ipynb     |

### 2. `gui_tools`

This library handles different libraries for gui widgets that may be used for GUI-based image processing.

| Script                   | Description                                                         | Examples                    |
| :----------------------- | :------------------------------------------------------------------ | --------------------------- |
| `BasicFletIMAQ.py`       | A very basic Flet-Python application to acquire and display images. | BasicFletIMAQ_example.ipynb |
| `NapariImageAnalyzer.py` | A very Napari-based application for image analysis.                 | BasicFletIMAQ_example.ipynb |

---

`Author`: Jose Rico-Jimenez

---
