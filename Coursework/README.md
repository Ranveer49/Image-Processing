# Computer Vision & Digital Image Processing Coursework

A consolidated collection of image-processing coursework and homework implemented in Python using OpenCV, NumPy, SciPy, scikit-image, and Matplotlib.

## Covered techniques

- Grayscale image representation and 8-bit bit-plane decomposition/reconstruction
- Gaussian, salt-and-pepper, speckle, and Poisson noise modelling
- Image restoration using Gaussian, median, and Wiener filtering
- Quantitative evaluation using PSNR and SSIM
- Binary mathematical morphology using set-based operations
- Erosion, dilation, opening, and closing
- RGB/HSV/CMYK concepts and HSV-based color segmentation
- Hue-based recoloring
- Adaptive thresholding and connected-component filtering
- Telea image inpainting
- Sobel, Laplacian, and Canny edge detection
- Otsu thresholding and contour extraction
- Object measurements: area, perimeter, centroid, and circularity
- Parameterized binary/geometric image generation

## Repository layout

```text
coursework/
├── 01_image_representation/
│   └── bit_plane_analysis.py
├── 02_image_restoration/
│   └── noise_and_restoration.py
├── 03_morphology/
│   └── morphology_from_sets.py
├── 04_color_processing/
│   └── color_spaces_and_segmentation.py
├── 05_segmentation_restoration/
│   └── object_segmentation_inpainting.py
├── 06_edge_and_feature_analysis/
│   └── edges_contours_measurement.py
├── 07_image_generation/
│   └── parameterized_patterns.py
├── results/
├── sample_image.jpg
├── Q3.jpeg
├── umbrella.png
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the modules

Each Python script uses paths relative to its own module directory. Run it from the corresponding folder, for example:

```bash
cd coursework/02_image_restoration
python noise_and_restoration.py
```

Generated figures and intermediate outputs are written under `coursework/results/`.

## Portfolio note

This folder consolidates image-processing work developed through coursework and homework. The code has been organized into focused, reproducible modules and extended with quantitative edge/feature analysis.
