import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("../Q3.jpeg")
OUT = Path("../results/edge_feature_analysis")
OUT.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(INPUT))
if image is None:
    raise FileNotFoundError(f"Could not read {INPUT}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32))
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

laplacian = cv2.Laplacian(blur, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
canny = cv2.Canny(blur, 50, 150)

# Otsu segmentation for a reproducible contour-analysis example.
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
measurements = []

for idx, contour in enumerate(contours, start=1):
    area = cv2.contourArea(contour)
    if area < 100:
        continue
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    if M["m00"]:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter else 0
    measurements.append((idx, area, perimeter, cx, cy, circularity))
    cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite(str(OUT / "sobel.png"), sobel)
cv2.imwrite(str(OUT / "laplacian.png"), laplacian)
cv2.imwrite(str(OUT / "canny.png"), canny)
cv2.imwrite(str(OUT / "otsu_binary.png"), binary)
cv2.imwrite(str(OUT / "contours.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

print("id\tarea\tperimeter\tcentroid_x\tcentroid_y\tcircularity")
for row in measurements:
    print(f"{row[0]}\t{row[1]:.1f}\t{row[2]:.1f}\t{row[3]:.1f}\t{row[4]:.1f}\t{row[5]:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
items = [
    (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original"),
    (sobel, "Sobel magnitude"),
    (laplacian, "Laplacian"),
    (canny, "Canny"),
    (binary, "Otsu binary"),
    (annotated, "Contours + bounding boxes"),
]
for ax, (im, title) in zip(axes.ravel(), items):
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "edge_feature_comparison.png", dpi=250, bbox_inches="tight")
plt.close()
