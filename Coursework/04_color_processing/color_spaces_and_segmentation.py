import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("../umbrella.png")
OUT = Path("../results/color_processing")
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(INPUT))
if img is None:
    raise FileNotFoundError(f"Could not read {INPUT}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Broad warm-color mask; tune these values for a different image.
mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([179, 255, 255]))
mask = cv2.bitwise_or(mask1, mask2)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Demonstrate hue-based recoloring while preserving S and V.
results = []
for name, hue in [("blue", 110), ("green", 60), ("yellow", 30), ("magenta", 150)]:
    recolored_hsv = hsv.copy()
    recolored_hsv[..., 0][mask > 0] = hue
    recolored = cv2.cvtColor(recolored_hsv, cv2.COLOR_HSV2RGB)
    results.append((name, recolored))
    cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR))

cv2.imwrite(str(OUT / "segmentation_mask.png"), mask)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original")
axes[0, 1].imshow(mask, cmap="gray"); axes[0, 1].set_title("HSV mask")
axes[0, 2].axis("off")
for ax, (name, result) in zip(axes[1], results[:3]):
    ax.imshow(result); ax.set_title(f"Hue → {name}")
for ax in axes.ravel(): ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "color_processing.png", dpi=250, bbox_inches="tight")
plt.close()
