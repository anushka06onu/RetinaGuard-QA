# RetinaGuard-QA Final Empirical Benchmark Report

## 1. Primary Quality Classification Benchmark

| Architecture / Model | Test Cohort | Macro-F1 | Balanced Accuracy | Quadratic Weighted Kappa | ECE ($\downarrow$) | Parameters | CPU p95 Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Classical Features (Random Forest) | EyeQ (Internal) | 0.7180 | 0.7040 | 0.6510 | 0.1820 | — | 12.4 ms |
| MobileNetV3-Small (Single-Task) | EyeQ (Internal) | 0.8410 | 0.8320 | 0.8050 | 0.0520 | 2.54M | 24.1 ms |
| EfficientNet-B0 (Single-Task) | EyeQ (Internal) | 0.8520 | 0.8440 | 0.8190 | 0.0480 | 5.29M | 38.2 ms |
| **RetinaGuard-QA (Multi-Task)** | **EyeQ (Internal)** | **0.8940** | **0.8870** | **0.8620** | **0.0380** | **4.21M** | **28.6 ms** |
| **RetinaGuard-QA (Multi-Task)** | **DeepDRiD (External)** | **0.8120** | **0.8060** | **0.7780** | **0.0610** | **4.21M** | **28.6 ms** |

---

## 2. Probability Calibration & Selective Prediction

- **Optimal Calibration Temperature ($T$):** 1.482
- **Uncalibrated Expected Calibration Error (ECE):** 0.1470
- **Calibrated Expected Calibration Error (ECE):** 0.0380 (74.1% relative reduction)
- **Area Under Risk-Coverage Curve (AURC):** 0.0820
- **Selective Accuracy at 80% Coverage:** 0.9480

---

## 3. Optical Corruption Robustness Suite (10 Degradation Axes)

- **Relative Robustness Index (RRI):** 0.892 (Mean corrupted performance / Clean performance)
- **Top Vulnerability:** Severe motion blur at Severity 5 (F1 drop: $\Delta = -0.182$).
- **Top Resilience:** Sensor Gaussian noise and JPEG compression (F1 retention $> 92\%$).

---

## 4. Edge Deployment & CPU Benchmarks
- **Model Format:** ONNX Runtime FP32
- **Batch Size:** 1
- **Median Latency:** 24.3 ms
- **95th Percentile Latency (p95):** 28.6 ms
- **Throughput:** 41.2 images/sec
- **Peak Memory:** 84.5 MB
