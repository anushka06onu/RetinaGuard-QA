"""Actionable decision policy and feedback formulation matching Phase 12 and 15."""

from typing import Any, Dict, List, Literal, Optional, cast

from .schemas import (
    DecisionAction,
    PredictionResponse,
    QualityAttributes,
    QualityProbabilities,
)


class DecisionPolicyEngine:
    """Triage decision engine enforcing uncertainty bounds, modality gating, and actionable capture advice."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.85,  # bits
        ood_energy_threshold: float = 1.0,
        ood_direction: str = "lower_is_ood",
        energy_ood_enabled: bool = False,
        model_version: str = "1.0.1",
        supported_heads: Optional[List[str]] = None,
    ):
        if ood_direction not in ["lower_is_ood", "higher_is_ood"]:
            raise ValueError(
                f"Invalid ood_direction: {ood_direction}. Must be 'lower_is_ood' or 'higher_is_ood'."
            )
        self.uncertainty_threshold = uncertainty_threshold
        self.ood_energy_threshold = ood_energy_threshold
        self.ood_direction = ood_direction
        self.energy_ood_enabled = energy_ood_enabled
        self.model_version = model_version
        self.supported_heads = supported_heads

    def formulate_feedback(
        self,
        decision: DecisionAction,
        quality: str,
        clarity_val: Optional[int],
        artifact_val: Optional[int],
        field_def_val: Optional[int],
    ) -> List[str]:
        feedback = []
        if decision == DecisionAction.UNSUPPORTED_INPUT:
            return [
                "Invalid image format or non-fundus modality. Please submit a valid color retinal fundus image."
            ]

        if decision == DecisionAction.MANUAL_REVIEW:
            feedback.append(
                "High model uncertainty or borderline acquisition quality. Request qualified clinician review."
            )

        if clarity_val is not None and clarity_val >= 2:
            feedback.append(
                "Possible blur or focus defect detected. Stabilize and refocus camera before recapture."
            )

        if artifact_val is not None and artifact_val >= 2:
            feedback.append(
                "Possible acquisition artifact detected. Inspect the lens, alignment, and field before recapture."
            )

        if field_def_val is not None and field_def_val >= 2:
            feedback.append(
                "Possible field centering/definition defect. Adjust patient fixation before recapture."
            )

        if quality in {"reject", "poor_or_reject", "poor_reject"} and not feedback:
            feedback.append(
                "Overall technical acquisition quality is inadequate. "
                "Recapture the image with appropriate focus, illumination, "
                "centering, and optical alignment."
            )

        if not feedback:
            feedback.append(
                "The model classified this image as technically acceptable under its experimental quality-assessment protocol. Clinical suitability still requires qualified review."
            )

        return feedback

    def evaluate(
        self,
        probs: Dict[str, float],
        uncertainty: float,
        ood_score: float,
        is_valid_modality: bool = True,
        attributes_raw: Optional[Dict[str, Optional[int]]] = None,
        latency_ms: Optional[float] = None,
    ) -> PredictionResponse:
        pred_class = max(list(probs.keys()), key=lambda k: probs[k])
        cal_conf = probs[pred_class]

        # Attribute decodings (only if supported and present)
        artifact_code = attributes_raw.get("artifact") if attributes_raw else None
        clarity_code = attributes_raw.get("clarity") if attributes_raw else None
        field_def_code = attributes_raw.get("field_definition") if attributes_raw else None

        attr_decodings = {
            "artifact": (
                {0: "none", 1: "mild", 2: "severe"}.get(artifact_code)
                if artifact_code is not None
                else None
            ),
            "clarity": (
                {0: "high", 1: "moderate", 2: "low"}.get(clarity_code)
                if clarity_code is not None
                else None
            ),
            "field_definition": (
                {0: "adequate", 1: "incomplete", 2: "poor"}.get(field_def_code)
                if field_def_code is not None
                else None
            ),
        }

        # OOD determination based on direction (retained for diagnostics)
        if self.ood_direction == "higher_is_ood":
            is_ood = ood_score > self.ood_energy_threshold
        else:
            is_ood = ood_score < self.ood_energy_threshold

        is_ood_active = is_ood if self.energy_ood_enabled else False

        # Production decision hierarchy
        if not is_valid_modality:
            decision = DecisionAction.UNSUPPORTED_INPUT
        elif is_ood_active or uncertainty > self.uncertainty_threshold:
            decision = DecisionAction.MANUAL_REVIEW
        elif pred_class in ["reject", "poor_or_reject", "poor_reject"]:
            decision = DecisionAction.RECAPTURE
        elif pred_class == "usable":
            decision = DecisionAction.MANUAL_REVIEW
        else:
            decision = DecisionAction.ACCEPT

        feedback = self.formulate_feedback(
            decision,
            pred_class,
            clarity_code,
            artifact_code,
            field_def_code,
        )

        if "poor_or_reject" in probs or "poor_reject" in probs:
            p_good = round(probs.get("good", 0.0), 4)
            p_poor = round(1.0 - p_good, 4)
            prob_obj = QualityProbabilities(good=p_good, poor_or_reject=p_poor, usable=None, reject=None)
        elif "usable" in probs and "reject" in probs:
            p_good = round(probs.get("good", 0.0), 4)
            p_usable = round(probs.get("usable", 0.0), 4)
            p_reject = round(1.0 - p_good - p_usable, 4)
            if p_reject < 0.0:
                p_reject = 0.0
                p_usable = round(1.0 - p_good, 4)
            prob_obj = QualityProbabilities(good=p_good, poor_or_reject=None, usable=p_usable, reject=p_reject)
        else:
            p_good = round(probs.get("good", 0.0), 4)
            p_poor = round(1.0 - p_good, 4)
            prob_obj = QualityProbabilities(good=p_good, poor_or_reject=p_poor, usable=None, reject=None)

        return PredictionResponse(
            model_version=self.model_version,
            quality=cast(Literal["good", "poor_or_reject", "usable", "reject"], pred_class),
            probabilities=prob_obj,
            calibrated_confidence=round(cal_conf, 4),
            uncertainty=round(uncertainty, 4),
            ood_score=round(ood_score, 4),
            decision=decision,
            quality_attributes=QualityAttributes(
                artifact=cast(Any, attr_decodings.get("artifact")),
                clarity=cast(Any, attr_decodings.get("clarity")),
                field_definition=cast(Any, attr_decodings.get("field_definition")),
            ),
            feedback=feedback,
            disclaimer="Technical image-quality assessment only; not a clinical diagnosis or treatment recommendation.",
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
        )
