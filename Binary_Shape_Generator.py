import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# 1. Create a binary image containing the letter
# ---------------------------------------------------
# ---------------------------------------------------
# 1. Create a binary image containing the word
# ---------------------------------------------------

word = input("Enter a word: ").upper()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 8
thickness = 15

# Find the size of the text
(text_width, text_height), baseline = cv2.getTextSize(
    word,
    font,
    font_scale,
    thickness
)

# Add some margin around the text
margin = 40

# Create image according to the actual text size
img_width = text_width + 2 * margin
img_height = text_height + baseline + 2 * margin

img = np.zeros((img_height, img_width), dtype=np.uint8)

# Position text inside the image
x = margin
y = margin + text_height

cv2.putText(
    img,
    word,
    (x, y),
    font,
    font_scale,
    255,
    thickness,
    cv2.LINE_AA
)
# Convert to a clean binary image
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


# ---------------------------------------------------
# 2. Create a morphological erosion kernel
# ---------------------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)


# ---------------------------------------------------
# 3. Create different eroded versions
# ---------------------------------------------------

eroded_1 = cv2.erode(
    mask,
    kernel,
    iterations=2
)

eroded_2 = cv2.erode(
    mask,
    kernel,
    iterations=6
)

eroded_3 = cv2.erode(
    mask,
    kernel,
    iterations=10
)


# ---------------------------------------------------
# 4. Extract the outer contour
# ---------------------------------------------------

outer_line = cv2.subtract(
    mask,
    eroded_1
)


# ---------------------------------------------------
# 5. Extract a second parallel contour
# ---------------------------------------------------

inner_line = cv2.subtract(
    eroded_2,
    eroded_3
)


# ---------------------------------------------------
# 6. Combine both contours
# ---------------------------------------------------

parallel_lines = cv2.bitwise_or(
    outer_line,
    inner_line
)


# ---------------------------------------------------
# 7. Convert black background to white
#    and white lines to black
# ---------------------------------------------------

result = 255 - parallel_lines


# ---------------------------------------------------
# 8. Display everything
# ---------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask, cmap="gray")
plt.title("Binary Object")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(parallel_lines, cmap="gray")
plt.title("Parallel Contours")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result, cmap="gray")
plt.title("Final Black-on-White Image")
plt.axis("off")

plt.tight_layout()
plt.show()