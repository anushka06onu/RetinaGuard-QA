"""Pydantic schemas matching Blueprint Phase 18 and Phase 24."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionAction(str, Enum):
    ACCEPT = "accept"
    RECAPTURE = "recapture"
    MANUAL_REVIEW = "manual_review"
    UNSUPPORTED_INPUT = "unsupported_input"


class QualityProbabilities(BaseModel):
    good: float = Field(..., description="Calibrated probability of Good quality [0.0, 1.0]")
    usable: float = Field(..., description="Calibrated probability of Usable quality [0.0, 1.0]")
    reject: float = Field(..., description="Calibrated probability of Reject quality [0.0, 1.0]")


class QualityAttributes(BaseModel):
    artifact: Optional[str] = Field(None, description="Artifact severity: 'none', 'mild', 'severe'")
    clarity: Optional[str] = Field(None, description="Clarity score: 'high', 'moderate', 'low'")
    field_definition: Optional[str] = Field(None, description="Field definition: 'adequate', 'incomplete', 'poor'")


class PredictionResponse(BaseModel):
    model_version: str = Field("1.0.0", description="Model architecture & checkpoint version")
    quality: str = Field(..., description="Predicted canonical quality: 'good', 'usable', or 'reject'")
    probabilities: QualityProbabilities
    calibrated_confidence: float = Field(..., description="Calibrated top-1 probability confidence")
    uncertainty: float = Field(..., description="Predictive entropy in bits [0.0, 1.58]")
    ood_score: float = Field(..., description="Out-of-Distribution Energy score")
    decision: DecisionAction = Field(..., description="Triage decision: accept, recapture, manual_review, unsupported_input")
    quality_attributes: QualityAttributes
    feedback: List[str] = Field(..., description="Actionable physical and optical capture instructions")
    disclaimer: str = Field(
        "Technical image-quality assessment only; not a clinical diagnosis or treatment recommendation.",
        description="Non-diagnostic clinical boundary notice"
    )
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")


class PredictionRequest(BaseModel):
    image_base64: Optional[str] = None
