# System Architecture: RetinaGuard-QA

## 1. Pipeline Overview
RetinaGuard-QA processes retinal images through an upstream triage pipeline:

1. **Physical & Modality Gate (`src/ood/visual_gate.py`)**: Checks channel distribution, contrast, and aspect ratios to reject non-fundus modalities (e.g. chest X-rays, skin lesions, blank frames) with zero computational overhead.
2. **Canonical Preprocessing (`src/preprocessing/transforms.py`)**: Automatically detects the circular retinal boundary, eliminates black camera borders, applies square padding, and scales the image to standard $384 \times 384$ px tensor representation.
3. **Multi-Task Backbone (`src/models/multi_task_head.py`)**: Shared deep convolutional encoder (MobileNetV3 / EfficientNet-B0) extracting pooled features to feed 4 dedicated heads:
   - **Quality Grade Head**: Softmax distribution across `[Good, Usable, Reject]`.
   - **Actionable Defect Head**: Sigmoid multi-label logits across 6 optical defect classes.
   - **Quality Score Head**: Continuous visual quality index $[0.0, 100.0]$.
   - **Latent Projection Head**: Normalized $128$-d embedding for Mahalanobis OOD distance.
4. **Uncertainty Calibration & Gating (`src/uncertainty/calibration.py` & `src/inference/decision_engine.py`)**:
   - Temperature scaling aligns raw confidence scores with true empirical accuracy.
   - Shannon entropy thresholding triggers selective prediction (`MANUAL_REVIEW`) on high-risk borderline images.
5. **Decision & Feedback Formulation**: Translates quantitative predictions into actionable clinician and operator instructions (`ACCEPT`, `USABLE_WITH_WARNING`, `RECAPTURE_WITH_GUIDANCE`, `MANUAL_REVIEW`, `UNSUPPORTED_OOD`).
