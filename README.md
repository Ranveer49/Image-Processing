# Image Processing & Computer Vision

A collection of image processing and computer vision projects implemented in Python using OpenCV, NumPy, and Matplotlib.

The repository combines coursework-based implementations with independent image-processing projects, covering fundamental image representation, enhancement, restoration, morphology, color processing, segmentation, edge detection, feature analysis, and image generation.

---

## Projects

### 1. Binary Shape Generator

Generates binary images containing parameterized geometric shapes and patterns.

**Concepts:**
- Binary image representation
- Geometric primitives
- Pixel-level image generation
- Parameterized pattern generation

**Technologies:** Python, OpenCV, NumPy, Matplotlib

---

## Image Processing Coursework

The `Coursework/` directory contains implementations of core image-processing techniques.

### 2. Image Representation

**File:** `Coursework/01_image_representation/bit_plane_analysis.py`

Explores image representation through bit-plane decomposition.

**Concepts:**
- Grayscale image representation
- Binary representation of pixel intensity
- Bit-plane slicing
- Image information at different bit levels

---

### 3. Image Restoration

**File:** `Coursework/02_image_restoration/noise_and_restoration.py`

Studies image degradation and restoration using different noise models and filtering techniques.

**Concepts:**
- Image noise
- Noise modelling
- Spatial filtering
- Image restoration
- Comparison of restoration results

---

### 4. Mathematical Morphology

**File:** `Coursework/03_morphology/morphology_from_sets.py`

Implements morphological operations based on set-theoretic concepts.

**Concepts:**
- Structuring elements
- Erosion
- Dilation
- Opening
- Closing
- Binary morphology

---

### 5. Color Processing & Segmentation

**File:** `Coursework/04_color_processing/color_spaces_and_segmentation.py`

Explores different color representations and their application to image segmentation.

**Concepts:**
- RGB color space
- HSV color space
- Color-space conversion
- Color thresholding
- Segmentation masks

---

### 6. Object Segmentation & Inpainting

**File:** `Coursework/05_segmentation_restoration/object_segmentation_inpainting.py`

Combines segmentation and image restoration techniques for extracting objects and reconstructing missing regions.

**Concepts:**
- Object segmentation
- Binary masks
- Region extraction
- Image inpainting
- Morphological preprocessing

---

### 7. Edge, Contour & Feature Analysis

**File:** `Coursework/06_edge_and_feature_analysis/edges_contours_measurement.py`

Analyzes image structure using edges, contours, and geometric measurements.

**Concepts:**
- Edge detection
- Contour extraction
- Shape analysis
- Object measurement
- Geometric properties

---

### 8. Parameterized Image Generation

**File:** `Coursework/07_image_generation/parameterized_patterns.py`

Generates structured image patterns using mathematical and geometric parameters.

**Concepts:**
- Coordinate systems
- Geometric transformations
- Parameterized image generation
- Pixel-level image construction

---

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib

## Repository Structure

```text
Image-Processing/
│
├── Binary_Shape_Generator.py
├── README.md
│
└── Coursework/
    │
    ├── 01_image_representation/
    ├── 02_image_restoration/
    ├── 03_morphology/
    ├── 04_color_processing/
    ├── 05_segmentation_restoration/
    ├── 06_edge_and_feature_analysis/
    ├── 07_image_generation/
    │
    ├── README.md
    ├── requirements.txt
    ├── sample_image.jpg
    ├── umbrella.png
    └── Q3.jpeg
