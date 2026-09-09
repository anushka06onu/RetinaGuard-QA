"""High-fidelity synthetic corruption engine for retinal image quality and device-shift robustness benchmarking.

Supported Degradations (5 Severity Levels Each):
1. Defocus / Gaussian Blur
2. Motion Blur
3. Underexposure (Low Flash / Dim Sensor)
4. Overexposure (Flash Saturation)
5. Uneven Illumination / Vignetting
6. JPEG Compression
7. Additive Sensor Noise (Gaussian)
8. Color Temperature Shift (Chromatic Imbalance)
"""

import io
from typing import Dict, List, Union, Callable
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter


def apply_gaussian_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate optical defocus blur with severity in [1, 5]."""
    sigmas = [1.0, 2.5, 4.0, 6.0, 9.0]
    sigma = sigmas[min(max(severity - 1, 0), 4)]
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_motion_blur(image: Image.Image, severity: int = 1, angle: float = 45.0) -> Image.Image:
    """Simulate camera shake or patient micro-saccades during acquisition."""
    kernel_sizes = [5, 9, 15, 21, 31]
    k_size = kernel_sizes[min(max(severity - 1, 0), 4)]

    # Create linear motion blur kernel
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    center = k_size // 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(k_size):
        offset = i - center
        r = int(round(center + offset * sin_a))
        c = int(round(center + offset * cos_a))
        if 0 <= r < k_size and 0 <= c < k_size:
            kernel[r, c] = 1.0

    kernel /= max(1.0, kernel.sum())

    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    out_arr = np.zeros_like(img_arr)
    from scipy.signal import convolve2d
    for ch in range(3):
        out_arr[:, :, ch] = convolve2d(img_arr[:, :, ch], kernel, mode="same", boundary="symm")

    return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))


def apply_underexposure(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate low illumination or inadequate xenon flash."""
    factors = [0.80, 0.60, 0.45, 0.30, 0.15]
    factor = factors[min(max(severity - 1, 0), 4)]
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_overexposure(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate intense flash saturation resulting in loss of macular detail."""
    factors = [1.25, 1.50, 1.80, 2.20, 2.75]
    factor = factors[min(max(severity - 1, 0), 4)]
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_illumination_gradient(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate optical vignetting or unaligned camera axis."""
    intensities = [0.20, 0.35, 0.50, 0.65, 0.80]
    strength = intensities[min(max(severity - 1, 0), 4)]

    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w, _ = img_arr.shape

    # Construct linear or radial gradient mask
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    dist = np.sqrt(x ** 2 + y ** 2)
    gradient = 1.0 - strength * (dist / np.max(dist))
    gradient = np.clip(gradient, 0.1, 1.0)[:, :, None]

    out_arr = img_arr * gradient
    return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))


def apply_jpeg_compression(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate transmission compression blocking artifacts."""
    qualities = [80, 50, 30, 15, 5]
    q = qualities[min(max(severity - 1, 0), 4)]
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=q)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_sensor_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate low-light CMOS sensor noise."""
    sigmas = [5.0, 12.0, 22.0, 38.0, 60.0]
    sigma = sigmas[min(max(severity - 1, 0), 4)]
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, img_arr.shape)
    out_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out_arr)


def apply_color_temperature_shift(image: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate white balance shift (warm/cool chromatic cast)."""
    shifts = [15, 30, 50, 75, 100]
    shift = shifts[min(max(severity - 1, 0), 4)]
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    # Enhance Red, suppress Blue
    img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + shift, 0, 255)
    img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] - shift, 0, 255)
    return Image.fromarray(img_arr.astype(np.uint8))


class SyntheticCorruptionEngine:
    """Unified engine to apply parameterized optical degradations across test sets."""

    CORRUPTIONS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
        "gaussian_blur": apply_gaussian_blur,
        "motion_blur": apply_motion_blur,
        "underexposure": apply_underexposure,
        "overexposure": apply_overexposure,
        "illumination_gradient": apply_illumination_gradient,
        "jpeg_compression": apply_jpeg_compression,
        "sensor_noise": apply_sensor_noise,
        "color_shift": apply_color_temperature_shift,
    }

    @classmethod
    def apply(cls, image: Image.Image, corruption_name: str, severity: int = 1) -> Image.Image:
        """Apply a specific corruption at a designated severity [1-5]."""
        if corruption_name not in cls.CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption_name}'. Choose from: {list(cls.CORRUPTIONS.keys())}")
        return cls.CORRUPTIONS[corruption_name](image, severity)

    @classmethod
    def get_all_corruption_names(cls) -> List[str]:
        return list(cls.CORRUPTIONS.keys())
