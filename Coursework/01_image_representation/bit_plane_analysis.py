import cv2
import numpy as np
from pathlib import Path

INPUT = Path("../sample_image.jpg")
OUT = Path("../results/bit_planes")
OUT.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not read {INPUT}")

# Extract 8 binary bit planes.
planes = [((image >> bit) & 1) * 255 for bit in range(8)]
for bit, plane in enumerate(planes, start=1):
    cv2.imwrite(str(OUT / f"bit_plane_{bit}.png"), plane.astype(np.uint8))

# Progressive reconstruction from most significant to least significant bit.
reconstruction = np.zeros_like(image, dtype=np.uint16)
for bit in range(7, -1, -1):
    reconstruction += ((planes[bit] // 255).astype(np.uint16) << bit)
    cv2.imwrite(str(OUT / f"reconstruction_through_bit_{bit + 1}.png"), reconstruction.astype(np.uint8))

final = reconstruction.astype(np.uint8)
error_pixels = np.count_nonzero(cv2.absdiff(image, final))
print(f"Image shape: {image.shape}")
print(f"Non-zero reconstruction error pixels: {error_pixels}")
