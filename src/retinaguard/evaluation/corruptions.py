"""Controlled synthetic corruption benchmark suite (10 types x 5 severities) per Phase 14."""

import io
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.signal import convolve2d


def apply_gaussian_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    sigmas = [1.0, 2.5, 4.0, 6.0, 9.0]
    s = sigmas[min(max(severity - 1, 0), 4)]
    return image.filter(ImageFilter.GaussianBlur(radius=s))


def apply_defocus_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    sigmas = [1.5, 3.0, 5.0, 7.5, 11.0]
    s = sigmas[min(max(severity - 1, 0), 4)]
    return image.filter(ImageFilter.GaussianBlur(radius=s))


def apply_motion_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    kernel_sizes = [5, 9, 15, 21, 31]
    k_size = kernel_sizes[min(max(severity - 1, 0), 4)]
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        kernel[i, i] = 1.0
    kernel /= max(1.0, kernel.sum())

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    out = np.zeros_like(arr)
    for c in range(3):
        out[:, :, c] = convolve2d(arr[:, :, c], kernel, mode="same", boundary="symm")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_brightness_increase(image: Image.Image, severity: int = 1) -> Image.Image:
    factors = [1.2, 1.45, 1.75, 2.1, 2.6]
    f = factors[min(max(severity - 1, 0), 4)]
    return ImageEnhance.Brightness(image).enhance(f)


def apply_brightness_decrease(image: Image.Image, severity: int = 1) -> Image.Image:
    factors = [0.8, 0.6, 0.45, 0.3, 0.15]
    f = factors[min(max(severity - 1, 0), 4)]
    return ImageEnhance.Brightness(image).enhance(f)


def apply_contrast_reduction(image: Image.Image, severity: int = 1) -> Image.Image:
    factors = [0.8, 0.6, 0.4, 0.25, 0.1]
    f = factors[min(max(severity - 1, 0), 4)]
    return ImageEnhance.Contrast(image).enhance(f)


def apply_gamma_shift(image: Image.Image, severity: int = 1) -> Image.Image:
    gammas = [1.3, 1.6, 2.0, 2.5, 3.2]
    g = gammas[min(max(severity - 1, 0), 4)]
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.power(arr, g) * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_jpeg_compression(image: Image.Image, severity: int = 1) -> Image.Image:
    qualities = [80, 50, 30, 15, 5]
    q = qualities[min(max(severity - 1, 0), 4)]
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_sensor_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    sigmas = [5.0, 12.0, 22.0, 38.0, 60.0]
    s = sigmas[min(max(severity - 1, 0), 4)]
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, s, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_uneven_illumination(image: Image.Image, severity: int = 1) -> Image.Image:
    strengths = [0.2, 0.35, 0.5, 0.65, 0.8]
    s = strengths[min(max(severity - 1, 0), 4)]
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    dist = np.sqrt(x**2 + y**2)
    gradient = np.clip(1.0 - s * (dist / np.max(dist)), 0.1, 1.0)[:, :, None]
    return Image.fromarray(np.clip(arr * gradient, 0, 255).astype(np.uint8))


class SyntheticCorruptionSuite:
    CORRUPTIONS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
        "gaussian_blur": apply_gaussian_blur,
        "defocus_blur": apply_defocus_blur,
        "motion_blur": apply_motion_blur,
        "brightness_increase": apply_brightness_increase,
        "brightness_decrease": apply_brightness_decrease,
        "contrast_reduction": apply_contrast_reduction,
        "gamma_shift": apply_gamma_shift,
        "jpeg_compression": apply_jpeg_compression,
        "sensor_noise": apply_sensor_noise,
        "uneven_illumination": apply_uneven_illumination,
    }

    @classmethod
    def apply(cls, image: Image.Image, corruption_name: str, severity: int = 1) -> Image.Image:
        if corruption_name not in cls.CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption_name}'")
        return cls.CORRUPTIONS[corruption_name](image, severity)

    @classmethod
    def get_all_names(cls) -> List[str]:
        return list(cls.CORRUPTIONS.keys())


def apply_optical_corruption(image: Image.Image, name: str, severity: int = 1) -> Image.Image:
    return SyntheticCorruptionSuite.apply(image, name, severity)
