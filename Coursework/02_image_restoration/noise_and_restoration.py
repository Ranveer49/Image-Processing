import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path

INPUT = Path("../sample_image.jpg")
OUT = Path("../results/restoration")
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read {INPUT}")
original = img.astype(np.float64) / 255.0
rng = np.random.default_rng(42)

def metrics(restored):
    return (
        peak_signal_noise_ratio(original, restored, data_range=1.0),
        structural_similarity(original, restored, data_range=1.0),
    )

gaussian_noisy = np.clip(original + rng.normal(0, 0.08, original.shape), 0, 1)
gaussian_restored = np.clip(gaussian_filter(gaussian_noisy, sigma=1), 0, 1)

sp_noisy = original.copy()
r = rng.random(original.shape)
p = 0.05
sp_noisy[r < p / 2] = 1
sp_noisy[r > 1 - p / 2] = 0
sp_restored = median_filter(sp_noisy, size=3)

speckle_noisy = np.clip(original + original * rng.normal(0, 0.15, original.shape), 0, 1)
speckle_restored = np.clip(wiener(speckle_noisy, mysize=5), 0, 1)

scale = 30
poisson_noisy = np.clip(rng.poisson(original * scale) / scale, 0, 1)
poisson_restored = np.clip(gaussian_filter(poisson_noisy, sigma=1), 0, 1)

cases = [
    ("Gaussian", gaussian_noisy, gaussian_restored, "Gaussian filter"),
    ("Salt & pepper", sp_noisy, sp_restored, "Median filter"),
    ("Speckle", speckle_noisy, speckle_restored, "Wiener filter"),
    ("Poisson", poisson_noisy, poisson_restored, "Gaussian filter"),
]

print(f"{'Noise':<16}{'Restoration':<18}{'PSNR (dB)':>12}{'SSIM':>10}")
print("-" * 56)
for name, noisy, restored, method in cases:
    psnr, ssim = metrics(restored)
    print(f"{name:<16}{method:<18}{psnr:>12.3f}{ssim:>10.3f}")

fig, axes = plt.subplots(4, 3, figsize=(12, 14))
for row, (name, noisy, restored, method) in enumerate(cases):
    psnr, ssim = metrics(restored)
    axes[row, 0].imshow(original, cmap="gray")
    axes[row, 0].set_title("Original")
    axes[row, 1].imshow(noisy, cmap="gray")
    axes[row, 1].set_title(f"{name} noise")
    axes[row, 2].imshow(restored, cmap="gray")
    axes[row, 2].set_title(f"{method}\nPSNR={psnr:.2f} dB, SSIM={ssim:.3f}")
    for ax in axes[row]:
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "noise_restoration_comparison.png", dpi=250, bbox_inches="tight")
plt.close()
