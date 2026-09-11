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
                  └───────┬──────┬──────┬──────┬─────┬─────┘
                          │      │      │      │     │
       ┌──────────────────┘      │      │      │     └──────────────────┐
       ▼                         ▼      ▼      ▼                        ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────────┐ ┌────────────────┐
│ EyeQ Quality │ │ DeepDRiD OQ  │ │ Artifact │ │ Clarity (Blur) │ │ Field Def      │
│ Head (3-cls) │ │ Head (2-cls) │ │ (3 levels│ │ (3 levels)     │ │ (3 levels)     │
│ [G, U, R]    │ │ [Good, Rej]  │ │ 0, 1, 2) │ │ 0, 1, 2)       │ │ 0, 1, 2)       │
└──────┬───────┘ └──────┬───────┘ └────┬─────┘ └───────┬────────┘ └───────┬────────┘
       │                │              │               │                  │
       └────────────────┼──────────────┼───────────────┼──────────────────┘
                        ▼              ▼               ▼
                  ┌────────────────────────────────────────┐
                  │  Latent Projection Head (128-d Vector) │
                  │  Free Energy OOD Score Calculation     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Calibrated Decision Engine & Abstention│
                  │ (src/retinaguard/inference/decision)   │
                  │  - Post-hoc Temperature Scaling        │
                  │  - Shannon Predictive Entropy (bits)   │
                  │  - Energy-based OOD Thresholding       │
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
│   ├── adapters.py          # Standardized manifest generators for EyeQ & DeepDRiD
│   ├── datasets.py          # PyTorch MultiTask Dataset with valid-label masks
│   ├── preprocessing.py     # Fast circular FOV boundary detection and squaring
│   └── splitting.py         # Patient-isolated, stratified train/val/test splitters
├── models/
│   ├── baselines.py         # Classical texture/contrast IQ features (RF / LR)
│   └── multitask.py         # Shared backbone with 5 multi-task heads & latent projection
├── training/
│   ├── engine.py            # AMP training, multi-head loss, QWK/macro-F1 metrics
│   ├── losses.py            # Focal, cross-entropy, and ordinal loss functions
│   └── train.py             # Single-task / Multi-task campaign runner with full provenance
├── evaluation/
│   ├── bootstrap.py         # Patient-clustered 95% bootstrap confidence intervals
│   └── metrics.py           # Macro-F1, Balanced Accuracy, QWK, ECE, NLL, Brier Score
├── uncertainty/
│   ├── calibration.py       # Post-hoc Temperature Scaling optimizer on validation set
│   └── ood.py               # Energy score & Mahalanobis distance OOD calculators
├── inference/
│   ├── decision_engine.py   # Deterministic triage policy mapping scores to actions
│   └── predictor.py         # ONNX Runtime & PyTorch CPU transient inference engine
└── utils/
    ├── hashing.py           # Cryptographic SHA-256 manifest & file verifiers
    └── logging.py           # Structured JSON and console loggers
```

---

## 2. Multi-Task Model & Head Dimensions

The primary neural model (`RetinaGuardMultiTaskModel` in [src/retinaguard/models/multitask.py](../src/retinaguard/models/multitask.py)) extracts a 512-dimensional or 960-dimensional pooled embedding from a shared convolutional backbone (`mobilenetv3_large_100` or `efficientnet_b0`).

The shared representation feeds five dedicated task heads:

1. **EyeQ Quality Head (`quality_head`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Classes: `0: Good`, `1: Usable`, `2: Reject`
   - Primary quality classification task evaluated on EyeQ test cohort.
2. **DeepDRiD Overall Quality Head (`overall_quality_head`)**:
   - Dimension: `nn.Linear(D, 2)`
   - Classes: `0: Good`, `1: Poor/Reject` (Strictly binary matching official DeepDRiD annotations).
3. **Artifact Degradation Head (`artifact_head`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Ordinal Levels: `0: None/Minimal`, `1: Moderate`, `2: Severe`
4. **Clarity / Defocus Head (`clarity_head`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Ordinal Levels: `0: Sharp/Normal`, `1: Mild Blur`, `2: Severe Blur`
5. **Field Definition Head (`field_def_head`)**:
   - Dimension: `nn.Linear(D, 3)`
   - Ordinal Levels: `0: Standard Centering`, `1: Mild Truncation`, `2: Severe Misalignment`
6. **Latent Projection Head (`latent_head`)**:
   - Dimension: `nn.Linear(D, 128)`
   - $L_2$-normalized embedding utilized for free-energy calculation and out-of-distribution distance scoring.

---

## 3. Mathematical Loss Formulation

During multi-task training across heterogeneous datasets with missing label components, loss is computed over valid sample masks $m \in \{0, 1\}$:

$$\mathcal{L}_{\text{total}} = w_q \mathcal{L}_q + w_{oq} \mathcal{L}_{oq} + w_a \mathcal{L}_a + w_c \mathcal{L}_c + w_f \mathcal{L}_f$$

Where:
- $\mathcal{L}_q = \frac{1}{\sum m_q} \sum_{i=1}^N m_{q,i} \cdot \text{CE}(z_{q,i}, y_{q,i})$ (EyeQ 3-class nominal quality)
- $\mathcal{L}_{oq} = \frac{1}{\sum m_{oq}} \sum_{i=1}^N m_{oq,i} \cdot \text{CE}(z_{oq,i}, y_{oq,i})$ (DeepDRiD 2-class binary overall quality)
- $\mathcal{L}_a = \frac{1}{\sum m_a} \sum_{i=1}^N m_{a,i} \cdot \text{CE}(z_{a,i}, y_{a,i})$ (Artifact degradation level)
- $\mathcal{L}_c = \frac{1}{\sum m_c} \sum_{i=1}^N m_{c,i} \cdot \text{CE}(z_{c,i}, y_{c,i})$ (Clarity / focus level)
- $\mathcal{L}_f = \frac{1}{\sum m_f} \sum_{i=1}^N m_{f,i} \cdot \text{CE}(z_{f,i}, y_{f,i})$ (Field definition level)
- Default loss weights: $w_q = 1.0, w_{oq} = 0.5, w_a = 0.2, w_c = 0.2, w_f = 0.2$.

---

## 4. Checkpoint Selection Objective

Model checkpoints are selected on validation splits using a predefined, frozen composite score:

$$\mathcal{S}_{\text{val}} = 0.50 \cdot \text{F1}_{\text{eyeq}} + 0.20 \cdot \text{F1}_{\text{dd\_oq}} + 0.10 \cdot \text{QWK}_{\text{artifact}} + 0.10 \cdot \text{QWK}_{\text{clarity}} + 0.10 \cdot \text{QWK}_{\text{field}}$$

Where $\text{F1}$ denotes macro-averaged F1 score and $\text{QWK}$ denotes Quadratic Weighted Kappa.

---

## 5. Post-Hoc Uncertainty Calibration & Gating

1. **Temperature Scaling**: Logits $z$ are scaled by validation-fitted temperature $T > 0$:
   $$p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
   Applied strictly once at the predictor level to prevent double calibration.

2. **Predictive Entropy**:
   $$H(p) = - \sum_{k=1}^K p_k \log_2 (p_k)$$
   Scans with $H(p) > \tau_u$ (derived at fixed coverage on validation data) trigger `manual_review`.

3. **Free Energy OOD Detection**:
   $$E(x; T) = - T \cdot \log \sum_{k=1}^K \exp\left(\frac{z_k}{T}\right)$$
   Images with $E(x) > \tau_{\text{energy}}$ trigger `unsupported_input`.

---

## 6. Actionable Triage Decisions

The decision engine maps model outputs to four operational states:

| Decision | Criterion | Action |
| :--- | :--- | :--- |
| `accept` | $y = \text{good}$, $H(p) \le \tau_u$, $E \le \tau_E$ | Image satisfies technical quality criteria. |
| `recapture` | $y = \text{reject}$, $H(p) \le \tau_u$, $E \le \tau_E$ | High defect severity detected. Guidance emitted (e.g. refocus lens, adjust flash). |
| `manual_review` | $H(p) > \tau_u$ OR $y = \text{usable}$ | High predictive uncertainty near decision boundary. Operator inspection requested. |
| `unsupported_input` | $E > \tau_E$ OR Modality heuristic failure | Non-fundus or corrupt image rejected before diagnostic processing. |
