import cv2
import numpy as np
from pathlib import Path

OUT = Path("../results/image_generation")
OUT.mkdir(parents=True, exist_ok=True)

# Parameterized circular dot pattern from the assignment work.
W = H = 1024
cx, cy = W // 2, 560
spacing = 32
r_max = 13
sigma = 260
pattern_radius = 500
img = np.full((H, W), 255, dtype=np.uint8)

for j in range(-H // spacing, H // spacing + 1):
    for i in range(-W // spacing, W // spacing + 1):
        x = cx + i * spacing
        y = cy + j * spacing
        if not (0 <= x < W and 0 <= y < H):
            continue
        d = np.hypot(x - cx, y - cy)
        if d <= pattern_radius:
            radius = r_max * np.exp(-(d ** 2) / (2 * sigma ** 2))
            if radius >= 0.5:
                cv2.circle(img, (x, y), int(round(radius)), 0, -1)

cv2.imwrite(str(OUT / "gaussian_dot_pattern.png"), img)
print("Generated parameterized circular dot pattern.")
