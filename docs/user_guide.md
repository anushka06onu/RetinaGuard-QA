# RetinaGuard-QA Operator & Research User Guide

> **Important Research Disclaimer:** RetinaGuard-QA is a research prototype for technical fundus image-quality assessment. It is not clinically validated for diagnostic decision making.

## Operator & Research Workflow

### 1. Ingesting an Image
- Access the web interface at `http://localhost:5173` (or port configured in your environment).
- Upload a standard color retinal fundus photograph (supported formats: **JPEG** or **PNG**, up to 15MB).
- Or run one of the synthetic interface fixtures to verify system connectivity and badge rendering.

### 2. Interpreting Triage Decisions

The decision engine outputs one of four deterministic states:

| Decision Badge | Status Meaning | Recommended Action |
| :--- | :--- | :--- |
| **`accept`** | Adequate Technical Quality | Image satisfies technical criteria. |
| **`recapture`** | Significant Quality Degradation | Quality is inadequate due to optical/illumination defects. Review the feedback guidance (e.g. refocus lens, adjust flash intensity, reposition patient). |
| **`manual_review`** | High Predictive Uncertainty | Model predictive entropy is elevated near decision threshold. Human expert review recommended. |
| **`unsupported_input`** | Out-of-Distribution / Invalid | Input failed circular FOV detection, modality prechecks, or energy score threshold. |

### 3. Understanding Quantitative Metrics

- **Quality Grade:** Primary classification (`Good`, `Usable`, `Reject`) with calibrated confidence percentage.
- **Predictive Entropy (bits):** Shannon entropy $H(p) = -\sum p_k \log_2(p_k)$ quantifying model uncertainty.
- **Energy Metric:** Free energy score used for OOD input gating.
- **Acquisition Attributes:** Predicted degradation levels for Artifact, Clarity, and Field Definition.
- **Capture Guidance:** Actionable instructions derived directly from predicted attribute levels.
