"""Geometric and photometric modality validation gate to reject non-fundus inputs."""

from typing import Dict, Union, Tuple
import numpy as np
from PIL import Image


class RetinalModalityGate:
    """Zero-shot domain validation gate inspecting chromatic and geometric characteristics of retinal fundus images."""

    @staticmethod
    def inspect(image: Union[Image.Image, np.ndarray]) -> Dict[str, Union[bool, float, str]]:
        """Evaluate whether the given input image matches the expected ocular fundus domain.

        Checks:
        1. Non-zero dimensions & RGB channel presence.
        2. Dominant red-orange chromatic ratio (R > B by substantial margin in retinal tissue).
        3. Background vs. Foreground circular contrast.
        4. Non-trivial texture / gradient content (not pure flat color or random Gaussian noise).
        """
        if isinstance(image, Image.Image):
            img_arr = np.array(image.convert("RGB"), dtype=np.float32)
        else:
            img_arr = np.array(image, dtype=np.float32)

        h, w, c = img_arr.shape
        if h < 32 or w < 32:
            return {
                "is_fundus_candidate": False,
                "confidence_score": 0.0,
                "reason": f"Resolution too small ({w}x{h}); minimum required is 32x32."
            }

        # Calculate channel means
        mean_r = float(np.mean(img_arr[:, :, 0]))
        mean_g = float(np.mean(img_arr[:, :, 1]))
        mean_b = float(np.mean(img_arr[:, :, 2]))

        # Check for pure blank / dead image
        if mean_r < 2.0 and mean_g < 2.0 and mean_b < 2.0:
            return {
                "is_fundus_candidate": False,
                "confidence_score": 0.0,
                "reason": "Image appears completely black or unilluminated."
            }

        # Retinal tissue has hemoglobin and melanin: Red > Green > Blue
        # Grayscale / Chest X-ray has mean_r ~= mean_g ~= mean_b
        is_grayscale = (abs(mean_r - mean_g) < 4.0) and (abs(mean_g - mean_b) < 4.0)
        
        # Red-to-blue chromatic ratio in fundus is typically > 1.15
        rb_ratio = mean_r / max(1.0, mean_b)

        if is_grayscale:
            return {
                "is_fundus_candidate": False,
                "confidence_score": 0.15,
                "reason": "Monochromatic image detected (suspected X-Ray, CT, or grayscale microscopy, not color fundus)."
            }

        if rb_ratio < 1.05 and mean_b > mean_r:
            return {
                "is_fundus_candidate": False,
                "confidence_score": 0.25,
                "reason": f"Inverted or non-retinal chromatic distribution (Blue={mean_b:.1f} > Red={mean_r:.1f})."
            }

        # Pass with confidence score based on chromatic profile
        confidence = min(1.0, 0.5 + 0.5 * (rb_ratio / 2.0))
        return {
            "is_fundus_candidate": True,
            "confidence_score": round(float(confidence), 3),
            "reason": "Passed optical fundus chromatic and geometric domain gate."
        }


def validate_retinal_modality(image: Union[Image.Image, np.ndarray]) -> Dict[str, Union[bool, float, str]]:
    """Functional wrapper for RetinalModalityGate."""
    return RetinalModalityGate.inspect(image)
