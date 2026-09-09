# Research Methodology: RetinaGuard-QA

## 1. Problem Formulation
Medical AI systems routinely fail silently when presented with corrupted, degraded, or out-of-domain images. RetinaGuard-QA formulates automated quality assurance as a multi-objective decision problem:

$$\text{Input Image } x \longrightarrow \left[ p(y|x), H(p), S_{\text{energy}}(x), \mathbf{a}(x) \right] \longrightarrow \text{Action } \in \{\text{Accept, Recapture, Manual Review, Unsupported}\}$$

## 2. Masked Multi-Task Loss
Because training datasets possess heterogeneous label taxonomies, we train with a masked objective:

$$\mathcal{L} = \lambda_q m_q \mathcal{L}_{\text{CE}}(y_q, \hat{y}_q) + \lambda_a m_a \mathcal{L}_{\text{CE}}(y_a, \hat{y}_a) + \lambda_c m_c \mathcal{L}_{\text{CE}}(y_c, \hat{y}_c) + \lambda_f m_f \mathcal{L}_{\text{CE}}(y_f, \hat{y}_f)$$

where $m_i \in \{0, 1\}$ denotes label availability.

## 3. Post-Hoc Probability Calibration
Predicted logits $\mathbf{z}$ are scaled by an empirical temperature parameter $T > 0$ fitted to minimize negative log-likelihood on validation data:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

## 4. Selective Prediction & Abstention Policy
Selective risk $\mathcal{R}(c)$ is measured as a function of sample coverage $c$. Images with Shannon entropy $H(p) > \tau_{\text{entropy}}$ or out-of-distribution energy $S_{\text{energy}} < \tau_{\text{energy}}$ trigger selective abstention (`Manual review`).
