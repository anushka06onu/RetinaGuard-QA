# Research Methodology: RetinaGuard-QA

## 1. Problem Formulation
Automated quality assurance is formulated as a multi-objective decision problem combining multi-task classification, ordinal attribute prediction, probability calibration, and uncertainty-aware selective abstention:

$$\text{Input Image } x \longrightarrow \left[ p(y|x), H(p), E(x), \mathbf{a}(x) \right] \longrightarrow \text{Action } \in \{\text{accept}, \text{recapture}, \text{manual\_review}, \text{unsupported\_input}\}$$

---

## 2. Multi-Task Objective with Masked Supervision

Learning proceeds under a multi-task objective spanning binary overall quality and three granular clinical degradation attributes on DeepDRiD:

$$\mathcal{L}_{\text{total}} = w_{oq} \mathcal{L}_{oq} + w_a \mathcal{L}_a + w_c \mathcal{L}_c + w_f \mathcal{L}_f$$

### Task Specifications & Taxonomy:

1. **DeepDRiD Overall Quality ($\mathcal{L}_{oq}$)**:
   - Nominal 2-class binary cross-entropy over $\{\text{Good}: 0, \text{Poor/Reject}: 1\}$:
   $$\mathcal{L}_{oq} = \frac{1}{\sum m_{oq}} \sum_{i=1}^N m_{oq,i} \cdot \text{CE}(z_{oq,i}, y_{oq,i})$$
2. **Artifact Level ($\mathcal{L}_a$)**:
   - Ordinal 3-level evaluation $\{\text{Level 0: None/Minimal}, \text{Level 1: Moderate}, \text{Level 2: Severe}\}$ evaluated with Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE), and cross-entropy loss:
   $$\mathcal{L}_a = \frac{1}{\sum m_a} \sum_{i=1}^N m_{a,i} \cdot \text{CE}(z_{a,i}, y_{a,i})$$
3. **Clarity / Defocus Level ($\mathcal{L}_c$)**:
   - Ordinal 3-level evaluation $\{\text{Level 0: Normal/Sharp}, \text{Level 1: Mild Blur}, \text{Level 2: Severe Blur}\}$:
   $$\mathcal{L}_c = \frac{1}{\sum m_c} \sum_{i=1}^N m_{c,i} \cdot \text{CE}(z_{c,i}, y_{c,i})$$
4. **Field Definition Level ($\mathcal{L}_f$)**:
   - Ordinal 3-level evaluation $\{\text{Level 0: Standard Centering}, \text{Level 1: Mild Truncation}, \text{Level 2: Severe Misalignment}\}$:
   $$\mathcal{L}_f = \frac{1}{\sum m_f} \sum_{i=1}^N m_{f,i} \cdot \text{CE}(z_{f,i}, y_{f,i})$$

---

## 3. Post-Hoc Probability Calibration

Raw network logits $\mathbf{z}$ are calibrated via temperature scaling:

$$p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

Temperature $T = 1.3956$ is optimized on the validation partition via L-BFGS to minimize negative log-likelihood (NLL) without modifying classification accuracy or altering test set rankings.

---

## 4. Selective Prediction & Abstention Policy

Selective prediction uses predictive entropy $H(p) = - \sum_k p_k \log_2(p_k) \in [0.0, 1.0\text{ bit}]$ to quantify predictive uncertainty for binary overall quality. An empirical threshold $\tau_u = 0.9943\text{ bits}$ is fitted on validation data to route ambiguous scans near the decision boundary to `manual_review`.
