"""Shared CIFAR normalization and lightweight single-image preprocessing."""
import numpy as np
from PIL import Image

NORMALIZATION = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def prepare_image(image, dataset, resolution):
    image = image.convert("RGB")
    size = (resolution, resolution)
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    mean, std = np.asarray(NORMALIZATION[dataset], dtype=np.float32)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (pixels - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
