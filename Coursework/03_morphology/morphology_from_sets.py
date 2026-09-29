import cv2
import numpy as np
from pathlib import Path

OUT = Path("../results/morphology")
OUT.mkdir(parents=True, exist_ok=True)

# Binary test image containing separated and connected foreground regions.
image = np.zeros((120, 160), dtype=np.uint8)
cv2.rectangle(image, (20, 25), (90, 90), 255, -1)
cv2.circle(image, (120, 60), 25, 255, -1)
cv2.circle(image, (45, 55), 4, 255, -1)  # small isolated foreground object


def binary_to_set(binary):
    return {tuple(p) for p in np.argwhere(binary > 0)}


def set_to_binary(points, shape):
    out = np.zeros(shape, dtype=np.uint8)
    for y, x in points:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            out[y, x] = 255
    return out


def reflected_offsets(kernel):
    cy, cx = np.array(kernel.shape) // 2
    return [(y - cy, x - cx) for y, x in zip(*np.where(kernel > 0))]


def erode_set(A, kernel):
    offsets = reflected_offsets(kernel)
    result = set()
    for y, x in A:
        if all((y + dy, x + dx) in A for dy, dx in offsets):
            result.add((y, x))
    return result


def dilate_set(A, kernel):
    offsets = reflected_offsets(kernel)
    result = set()
    for y, x in A:
        for dy, dx in offsets:
            result.add((y + dy, x + dx))
    return result


def opening_set(A, kernel):
    return dilate_set(erode_set(A, kernel), kernel)


def closing_set(A, kernel):
    return erode_set(dilate_set(A, kernel), kernel)

A = binary_to_set(image)
kernel = np.ones((3, 3), dtype=np.uint8)
E = set_to_binary(erode_set(A, kernel), image.shape)
D = set_to_binary(dilate_set(A, kernel), image.shape)
O = set_to_binary(opening_set(A, kernel), image.shape)
C = set_to_binary(closing_set(A, kernel), image.shape)

cv2.imwrite(str(OUT / "original.png"), image)
cv2.imwrite(str(OUT / "erosion.png"), E)
cv2.imwrite(str(OUT / "dilation.png"), D)
cv2.imwrite(str(OUT / "opening.png"), O)
cv2.imwrite(str(OUT / "closing.png"), C)
print("Implemented erosion, dilation, opening and closing using binary sets.")
