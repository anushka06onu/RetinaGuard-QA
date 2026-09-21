# System Architecture: RetinaGuard-QA

RetinaGuard-QA is an uncertainty-aware, multi-task image quality assessment system designed for color retinal fundus photography. It executes lightweight, transient CPU inference, predicts granular acquisition attributes, estimates predictive entropy, detects out-of-distribution (OOD) inputs, and routes borderline scans via calibrated selective prediction.

```
                  ┌────────────────────────────────────────┐
                  │       Uploaded Retinal Fundus Image     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Modality & Heuristic Precheck     │
                  │   (src/retinaguard/data/preprocessing) │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Canonical FOV Alignment & Crop   │
                  │     (384x384 px Normalized Tensor)     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Shared Deep Convolutional Backbone │
                  │  (MobileNetV3-Large / EfficientNet-B0) │
                  │     (src/retinaguard/models/multitask) │
                  └───────┬──────────┬──────────┬──────────┘
                          │          │          │          
                          ▼          ▼          ▼          
                  ┌──────────────┐ ┌──────────┐ ┌────────────────┐ ┌────────────────┐
                  │ DeepDRiD OQ  │ │ Artifact │ │ Clarity (Blur) │ │ Field Def      │
                  │ Head (2-cls) │ │ (3 levels│ │ (3 levels)     │ │ (3 levels)     │
                  │ [Good, Rej]  │ │ 0, 1, 2) │ │ 0, 1, 2)       │ │ 0, 1, 2)       │
                  └──────┬───────┘ └────┬─────┘ └───────┬────────┘ └───────┬────────┘
                         │              │               │                  │
                         └──────────────┼───────────────┼──────────────────┘
                                        ▼               ▼
                  ┌────────────────────────────────────────┐
                  │ Calibrated Decision Engine & Abstention│
                  │ (src/retinaguard/inference/decision)   │
                  │  - Post-hoc Temperature Scaling (1.39) │
                  │  - Shannon Predictive Entropy (bits)   │
                  │  - Research OOD Energy Score Logging   │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Triage Decision:                       │
                  │ [accept | recapture | manual_review    │
                  │  | unsupported_input]                  │
                  │ + Actionable Acquisition Guidance      │
                  └────────────────────────────────────────┘
```

---

## 1. Modular Directory Structure

All runtime code is organized within the `retinaguard` Python package namespace:

```text
src/retinaguard/
├── data/
│   ├── adapters.py          # Standardized manifest generators for DeepDRiD
│   ├── datasets.py          # PyTorch MultiTask Dataset with valid-label masks
│   ├── preprocessing.py     # Fast circular FOV boundary detection and squaring
│   └── splitting.py         # Patient-isolated, stratified train/val/test splitters
├── models/
│   ├── baselines.py         # Classical texture/contrast IQ features (RF / LR)
│   └── multitask.py         # Shared backbone with multi-task classification heads
├── training/
│   ├── engine.py            # AMP training, multi-head loss, QWK/macro-F1 metrics
│   ├── losses.py            # Cross-entropy and ordinal loss functions
│   └── train.py             # Multi-task campaign runner with full provenance
├── evaluation/
│   ├── bootstrap.py         # Patient-clustered 95% bootstrap confidence intervals
│   └── metrics.py           # Macro-F1, Balanced Accuracy, QWK, ECE, NLL, Brier Score
├── uncertainty/
│   ├── calibration.py       # Post-hoc Temperature Scaling optimizer on validation set
│   └── ood.py               # Energy score OOD research calculator
├── inference/
│   ├── decision_engine.py   # Deterministic triage policy mapping scores to actions
│   └── predictor.py         # ONNX Runtime & PyTorch CPU transient inference engine
└── utils/
    ├── hashing.py           # Cryptographic SHA-256 manifest & file verifiers
    └── logging.py           # Structured JSON and console loggers
```

---

## 2. Multi-Task Model & Head Dimensions

The primary neural model (`RetinaGuardMultiTaskModel` in [src/retinaguard/models/multitask.py](../src/retinaguard/models/multitask.py)) extracts a 960-dimensional pooled embedding from a shared convolutional backbone (`mobilenetv3_large_100`).

The shared representation feeds four dedicated task heads:

1. **DeepDRiD Overall Quality Head (`overall_quality_logits`)**:
   - Dimension: `nn.Linear(D, 2)`
   - Classes: `0: Good`, `1: Poor/Reject` (Strictly binary matching official DeepDRiD annotations).
   - Primary deployed quality classification head.
2. **Artifact Degradation Head (`artifact_logits`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Ordinal Levels: `0: None/Minimal`, `1: Moderate`, `2: Severe`
3. **Clarity / Defocus Head (`clarity_logits`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Ordinal Levels: `0: Sharp/Normal`, `1: Mild Blur`, `2: Severe Blur`
4. **Field Definition Head (`field_definition_logits`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Ordinal Levels: `0: Standard Centering`, `1: Mild Truncation`, `2: Severe Misalignment`

---

## 3. Mathematical Loss Formulation

During multi-task training across the four DeepDRiD task heads, the joint loss is computed over valid sample masks $m \in \{0, 1\}$:

$$\mathcal{L}_{\text{total}} = w_{oq} \mathcal{L}_{oq} + w_a \mathcal{L}_a + w_c \mathcal{L}_c + w_f \mathcal{L}_f$$

Where:
- $\mathcal{L}_{oq} = \frac{1}{\sum m_{oq}} \sum_{i=1}^N m_{oq,i} \cdot \text{CE}(z_{oq,i}, y_{oq,i})$ (DeepDRiD 2-class binary overall quality)
- $\mathcal{L}_a = \frac{1}{\sum m_a} \sum_{i=1}^N m_{a,i} \cdot \text{CE}(z_{a,i}, y_{a,i})$ (Artifact degradation level)
- $\mathcal{L}_c = \frac{1}{\sum m_c} \sum_{i=1}^N m_{c,i} \cdot \text{CE}(z_{c,i}, y_{c,i})$ (Clarity / focus level)
- $\mathcal{L}_f = \frac{1}{\sum m_f} \sum_{i=1}^N m_{f,i} \cdot \text{CE}(z_{f,i}, y_{f,i})$ (Field definition level)
- Default loss weights: $w_{oq} = 1.0, w_a = 0.5, w_c = 0.5, w_f = 0.5$.

---

## 4. Checkpoint Selection Objective

Model checkpoints are selected on validation splits using the primary evaluation metric:

$$\mathcal{S}_{\text{val}} = \text{Macro-F1}_{\text{overall\_quality}}$$

with secondary tracking of Quadratic Weighted Kappa ($\text{QWK}$) across the three granular attribute heads.

---

## 5. Post-Hoc Uncertainty Calibration & Gating

1. **Temperature Scaling**: Logits $z$ are scaled by validation-fitted temperature $T = 1.3956$:
   $$p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
   Applied strictly once at the predictor level to prevent double calibration.

2. **Predictive Entropy**:
   $$H(p) = - \sum_{k=1}^2 p_k \log_2 (p_k) \in [0.0, 1.0\text{ bit}]$$
   Scans with $H(p) > 0.9943\text{ bits}$ (derived from validation percentile calibration) trigger `manual_review`.

3. **Free Energy OOD Logging**:
   $$E(x; T) = - T \cdot \log \sum_{k=1}^K \exp\left(\frac{z_k}{T}\right)$$
   Retained and logged for research and distribution analysis.

---

## 6. Actionable Triage Decisions

The decision engine maps model outputs to four operational states:

| Decision | Criterion | Action |
| :--- | :--- | :--- |
| `accept` | $y = \text{good}$, $H(p) \le \tau_u$, Modality Valid | Image satisfies technical quality criteria. |
| `recapture` | $y = \text{poor\_or\_reject}$, $H(p) \le \tau_u$, Modality Valid | High defect severity detected. Actionable guidance emitted (e.g. refocus, adjust flash, realign gaze). |
| `manual_review` | $H(p) > \tau_u$ ($0.9943\text{ bits}$) | High predictive uncertainty near decision boundary. Operator inspection requested. |
| `unsupported_input` | Modality heuristic failure / Non-fundus | Corrupt or non-retinal image rejected before diagnostic processing. |
