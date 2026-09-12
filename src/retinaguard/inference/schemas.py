"""Pydantic schemas matching Blueprint Phase 18 and Phase 24."""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class DecisionAction(str, Enum):
    ACCEPT = "accept"
    RECAPTURE = "recapture"
    MANUAL_REVIEW = "manual_review"
    UNSUPPORTED_INPUT = "unsupported_input"


class QualityProbabilities(BaseModel):
    good: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of Good quality [0.0, 1.0]")
    usable: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of Usable quality [0.0, 1.0]")
    reject: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of Reject quality [0.0, 1.0]")

    @model_validator(mode="after")
    def check_sum(self) -> "QualityProbabilities":
        total = self.good + self.usable + self.reject
        if not (0.90 <= total <= 1.10):
            raise ValueError(f"Probabilities must approximately sum to 1.0, got {total}")
        return self


class QualityAttributes(BaseModel):
    artifact: Optional[Literal["none", "mild", "severe"]] = Field(None, description="Artifact severity: 'none', 'mild', 'severe'")
    clarity: Optional[Literal["high", "moderate", "low"]] = Field(None, description="Clarity score: 'high', 'moderate', 'low'")
    field_definition: Optional[Literal["adequate", "incomplete", "poor"]] = Field(
        None, description="Field definition: 'adequate', 'incomplete', 'poor'"
    )


class PredictionResponse(BaseModel):
    model_version: str = Field("0.2.0", description="Model architecture & checkpoint version")
    quality: Literal["good", "usable", "reject"] = Field(
        ..., description="Predicted canonical quality: 'good', 'usable', or 'reject'"
    )
    probabilities: QualityProbabilities
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated top-1 probability confidence")
    uncertainty: float = Field(..., ge=0.0, le=1.60, description="Predictive entropy in bits [0.0, 1.585]")
    ood_score: float = Field(..., description="Out-of-Distribution Energy score")
    decision: DecisionAction = Field(
        ..., description="Triage decision: accept, recapture, manual_review, unsupported_input"
    )
    quality_attributes: QualityAttributes
    feedback: List[str] = Field(
        ..., description="Actionable physical and optical capture instructions"
    )
    disclaimer: str = Field(
        "Technical image-quality assessment only; not a clinical diagnosis or treatment recommendation.",
        description="Non-diagnostic clinical boundary notice",
    )
    latency_ms: Optional[float] = Field(None, ge=0.0, description="Inference latency in milliseconds")


class PredictionRequest(BaseModel):
    image_base64: Optional[str] = None

