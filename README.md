# Video Comparison Script

This Python script compares two videos frame by frame using three similarity metrics: Structural Similarity Index (SSIM), Mean Squared Error (MSE), and histogram similarity. It helps in analyzing the similarity between two video files. 

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Installation](#installation)

## Features

- Computes SSIM to measure structural similarity between frames.
- Calculates Mean Squared Error (MSE) for pixel-wise differences.
- Measures histogram similarity using correlation.
- Reports average similarity metrics and counts frames above a defined SSIM threshold.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- scikit-image (`scikit-image`)

## Usage

To compare two video files, insert two video files and name it video1.mp4 and video2.mp4

## Installation

To install the required libraries, you can use pip:

```bash
pip install opencv-python numpy scikit-image




