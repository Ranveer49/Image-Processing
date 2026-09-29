import cv2
import numpy as np
from pathlib import Path

OUT = Path("../results/segmentation_restoration")
OUT.mkdir(parents=True, exist_ok=True)


def remove_colored_objects(input_path, output_name, lower_ranges, upper_ranges):
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {input_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in zip(lower_ranges, upper_ranges):
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, open_kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            clean[labels == i] = 255

    result = cv2.inpaint(image, clean, 7, cv2.INPAINT_TELEA)
    cv2.imwrite(str(OUT / f"{output_name}_mask.png"), clean)
    cv2.imwrite(str(OUT / f"{output_name}_result.png"), result)
    return image, clean, result

# Uses the same HSV-based strategy as the assignment's leaf/twig-removal task.
remove_colored_objects(
    Path("../Q3.jpeg"),
    "leaf_twig_removal",
    [(15, 70, 60), (5, 90, 50)],
    [(95, 255, 255), (25, 255, 230)],
)

print("Segmentation → morphology → connected components → Telea inpainting completed.")
