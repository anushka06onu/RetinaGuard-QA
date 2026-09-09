"""Color constancy, local space normalization (Graham's method), and CLAHE enhancement."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


from typing import Optional, Union

def graham_color_normalization(
    image: Optional[Union[Image.Image, np.ndarray]] = None,
    image_np: Optional[np.ndarray] = None,
    scale: float = 300.0,
    sigma: float = 10.0
) -> Image.Image:
    """Ben Graham's local color space subtraction method (Kaggle Diabetic Retinopathy Winner).

    Formula: I_norm = alpha * I + beta * GaussianBlur(I) + gamma
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    elif image_np is not None:
        img_arr = np.array(image_np, dtype=np.float32)
    elif isinstance(image, np.ndarray):
        img_arr = np.array(image, dtype=np.float32)
    else:
        raise ValueError("Must provide either a PIL Image or NumPy array")

    # Local average background estimation per channel
    blurred = np.zeros_like(img_arr)
    for c in range(3):
        blurred[:, :, c] = gaussian_filter(img_arr[:, :, c], sigma=sigma)

    # Blend: 4 * img - 4 * blurred + 128
    norm = 4.0 * img_arr - 4.0 * blurred + 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    return Image.fromarray(norm)


def apply_clahe_enhancement(
    image: Image.Image, 
    clip_limit: float = 2.0, 
    grid_size: int = 8
) -> Image.Image:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel in LAB space."""
    img_arr = np.array(image.convert("RGB"))
    # Simple lightness channel adaptive equalization
    try:
        from skimage import color, exposure
        lab = color.rgb2lab(img_arr)
        l_chan = lab[:, :, 0] / 100.0
        l_clahe = exposure.equalize_adapthist(l_chan, clip_limit=clip_limit / 100.0, nbins=256)
        lab[:, :, 0] = l_clahe * 100.0
        rgb = color.lab2rgb(lab)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)
    except ImportError:
        # Fallback if skimage is not available: channel-wise percentile stretch
        p2, p98 = np.percentile(img_arr, (2, 98))
        rescaled = np.clip((img_arr - p2) / max(1e-5, p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(rescaled)


def reinhard_color_transfer(
    source: Image.Image, 
    target_mean: np.ndarray = np.array([140.0, 65.0, 25.0]), 
    target_std: np.ndarray = np.array([45.0, 18.0, 12.0])
) -> Image.Image:
    """Reinhard color normalization across RGB channels to match target fundus statistical distribution."""
    src_arr = np.array(source.convert("RGB"), dtype=np.float32)
    src_mean = src_arr.mean(axis=(0, 1))
    src_std = src_arr.std(axis=(0, 1)) + 1e-6

    norm = ((src_arr - src_mean) / src_std) * target_std + target_mean
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return Image.fromarray(norm)
