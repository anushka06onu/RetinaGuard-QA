"""Actionable triage and capture-feedback decision rule engine."""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class DecisionAction(str, Enum):
    ACCEPT = "ACCEPT"
    USABLE_WITH_WARNING = "USABLE_WITH_WARNING"
    RECAPTURE_WITH_GUIDANCE = "RECAPTURE_WITH_GUIDANCE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    UNSUPPORTED_OOD = "UNSUPPORTED_OOD"


@dataclass
class QualityAssessmentResult:
    """Structured quality assessment and actionable feedback payload."""
    decision: DecisionAction
    action_title: str
    action_summary: str
    quality_grade: str  # "Good", "Usable", "Reject"
    quality_grade_confidence: float
    grade_probabilities: Dict[str, float]
    quality_score: float  # [0, 100]
    is_usable_for_clinical_review: bool
    requires_recapture: bool
    requires_manual_review: bool
    is_out_of_distribution: bool
    predictive_entropy: float
    detected_defects: List[Dict[str, float]]
    actionable_instructions: List[str]
    latency_ms: float
    model_version: str = "1.0.0"
    clinical_disclaimer: str = (
        "RetinaGuard-QA assesses technical acquisition quality only. "
        "It does not diagnose pathology or replace qualified clinical evaluation."
    )

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


class DecisionEngine:
    """Evaluates multi-task quality logits, defect probabilities, and uncertainty metrics to formulate concrete operator directives."""

    DEFECT_FEEDBACK_MAP = {
        0: {
            "name": "Severe Blur / Focus Loss",
            "instruction": "Refocus the camera with emphasis on sharp retinal vessel and optic disc margins before recapture."
        },
        1: {
            "name": "Underexposure",
            "instruction": "Increase xenon flash intensity setting or check patient pupil dilation status."
        },
        2: {
            "name": "Overexposure",
            "instruction": "Reduce flash energy level to prevent saturation artifacts across the macular region."
        },
        3: {
            "name": "Uneven Illumination / Shadowing",
            "instruction": "Re-align optical working distance and ensure the camera objective is centered directly over the pupil."
        },
        4: {
            "name": "Field Truncation / Off-Center",
            "instruction": "Instruct the patient to steadily fixate on the target star to position the fovea centrally."
        },
        5: {
            "name": "Optical Artifacts / Lens Smear",
            "instruction": "Clean front objective lens with a lint-free optical tissue and ensure eyelashes are kept clear."
        }
    }

    def __init__(
        self,
        entropy_threshold: float = 0.85,
        energy_threshold: float = -3.5,
        defect_threshold: float = 0.50,
        grade_names: List[str] = ["Good", "Usable", "Reject"]
    ):
        self.entropy_threshold = entropy_threshold
        self.energy_threshold = energy_threshold
        self.defect_threshold = defect_threshold
        self.grade_names = grade_names

    def evaluate(
        self,
        grade_probs: np.ndarray,
        defect_probs: np.ndarray,
        quality_score: float,
        energy_score: float,
        entropy: float,
        is_fundus_modality: bool = True,
        latency_ms: float = 0.0
    ) -> QualityAssessmentResult:
        """Execute decision tree logic."""
        pred_grade_idx = int(np.argmax(grade_probs))
        pred_grade = self.grade_names[pred_grade_idx]
        conf = float(grade_probs[pred_grade_idx])

        grade_prob_dict = {
            self.grade_names[i]: float(grade_probs[i])
            for i in range(len(self.grade_names))
        }

        # 1. Gate 1: Non-fundus modality or extreme OOD energy
        if not is_fundus_modality or energy_score < self.energy_threshold:
            return QualityAssessmentResult(
                decision=DecisionAction.UNSUPPORTED_OOD,
                action_title="Unsupported Image / Out-of-Distribution",
                action_summary="The input image is not recognized as a valid color fundus photograph.",
                quality_grade="Reject",
                quality_grade_confidence=conf,
                grade_probabilities=grade_prob_dict,
                quality_score=0.0,
                is_usable_for_clinical_review=False,
                requires_recapture=True,
                requires_manual_review=True,
                is_out_of_distribution=True,
                predictive_entropy=float(entropy),
                detected_defects=[],
                actionable_instructions=[
                    "Verify image modality: please upload a standard non-mydriatic or mydriatic color fundus photograph."
                ],
                latency_ms=latency_ms
            )

        # 2. Extract detected actionable defects
        detected_defects = []
        actionable_instructions = []

        for idx, def_prob in enumerate(defect_probs):
            if def_prob >= self.defect_threshold:
                info = self.DEFECT_FEEDBACK_MAP.get(idx, {"name": f"Defect {idx}", "instruction": "Check camera."})
                detected_defects.append({
                    "name": info["name"],
                    "probability": round(float(def_prob), 3)
                })
                actionable_instructions.append(info["instruction"])

        # 3. Gate 2: High Predictive Uncertainty (Selective Abstention)
        if entropy > self.entropy_threshold:
            return QualityAssessmentResult(
                decision=DecisionAction.MANUAL_REVIEW,
                action_title="Manual Review Required (High Uncertainty)",
                action_summary="The automated quality model is uncertain. Human operator review is required.",
                quality_grade=pred_grade,
                quality_grade_confidence=conf,
                grade_probabilities=grade_prob_dict,
                quality_score=round(float(quality_score), 1),
                is_usable_for_clinical_review=False,
                requires_recapture=False,
                requires_manual_review=True,
                is_out_of_distribution=False,
                predictive_entropy=float(entropy),
                detected_defects=detected_defects,
                actionable_instructions=actionable_instructions or [
                    "Perform visual inspection of macula and optic disc clarity before proceeding."
                ],
                latency_ms=latency_ms
            )

        # 4. Gate 3: Reject Grade -> Recapture
        if pred_grade == "Reject":
            return QualityAssessmentResult(
                decision=DecisionAction.RECAPTURE_WITH_GUIDANCE,
                action_title="Recapture Required",
                action_summary="Image quality is inadequate for reliable clinical diagnosis.",
                quality_grade="Reject",
                quality_grade_confidence=conf,
                grade_probabilities=grade_prob_dict,
                quality_score=round(float(quality_score), 1),
                is_usable_for_clinical_review=False,
                requires_recapture=True,
                requires_manual_review=False,
                is_out_of_distribution=False,
                predictive_entropy=float(entropy),
                detected_defects=detected_defects,
                actionable_instructions=actionable_instructions or [
                    "Recapture image ensuring correct optical alignment and patient fixation."
                ],
                latency_ms=latency_ms
            )

        # 5. Gate 4: Usable Grade -> Usable with Warning
        if pred_grade == "Usable":
            return QualityAssessmentResult(
                decision=DecisionAction.USABLE_WITH_WARNING,
                action_title="Usable with Advisory Warning",
                action_summary="Image contains minor technical imperfections but is sufficient for clinical review.",
                quality_grade="Usable",
                quality_grade_confidence=conf,
                grade_probabilities=grade_prob_dict,
                quality_score=round(float(quality_score), 1),
                is_usable_for_clinical_review=True,
                requires_recapture=False,
                requires_manual_review=False,
                is_out_of_distribution=False,
                predictive_entropy=float(entropy),
                detected_defects=detected_defects,
                actionable_instructions=actionable_instructions or [
                    "Proceed to clinical review with attention to peripheral regions."
                ],
                latency_ms=latency_ms
            )

        # 6. Gate 5: Good Grade -> Accept
        return QualityAssessmentResult(
            decision=DecisionAction.ACCEPT,
            action_title="Accept (High Quality)",
            action_summary="Image demonstrates excellent clarity, illumination, and anatomical visibility.",
            quality_grade="Good",
            quality_grade_confidence=conf,
            grade_probabilities=grade_prob_dict,
            quality_score=round(float(quality_score), 1),
            is_usable_for_clinical_review=True,
            requires_recapture=False,
            requires_manual_review=False,
            is_out_of_distribution=False,
            predictive_entropy=float(entropy),
            detected_defects=detected_defects,
            actionable_instructions=["Ready for diagnostic pipeline / clinician review."],
            latency_ms=latency_ms
        )
