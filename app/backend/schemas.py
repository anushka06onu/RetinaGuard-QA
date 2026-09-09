"""Pydantic schemas for RetinaGuard-QA REST API endpoints."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DefectDetail(BaseModel):
    name: str = Field(..., description="Name of the detected acquisition defect")
    probability: float = Field(..., description="Estimated presence probability [0.0, 1.0]")


class QualityAssessmentResponse(BaseModel):
    decision: str = Field(..., description="Triage action (ACCEPT, USABLE_WITH_WARNING, RECAPTURE_WITH_GUIDANCE, MANUAL_REVIEW, UNSUPPORTED_OOD)")
    action_title: str
    action_summary: str
    quality_grade: str = Field(..., description="'Good', 'Usable', or 'Reject'")
    quality_grade_confidence: float
    grade_probabilities: Dict[str, float]
    quality_score: float = Field(..., description="Normalized visual fidelity index in [0.0, 100.0]")
    is_usable_for_clinical_review: bool
    requires_recapture: bool
    requires_manual_review: bool
    is_out_of_distribution: bool
    predictive_entropy: float
    detected_defects: List[DefectDetail]
    actionable_instructions: List[str]
    latency_ms: float
    model_version: str = "1.0.0"
    clinical_disclaimer: str


class HealthCheckResponse(BaseModel):
    status: str
    system: str
    version: str
    onnx_available: bool
    runtime_device: str
