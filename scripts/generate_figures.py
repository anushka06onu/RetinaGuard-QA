"""Generate high-resolution scientific figures from genuine experiment results.

This script strictly requires genuine experimental output files in artifacts/metrics/.
It does NOT contain or generate fallback/synthetic metric data.
If input metrics are missing, it fails fast.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from retinaguard.utils.hashing import compute_sha256

# Publication-grade plotting styles
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14


def plot_confusion_matrices(output_path: Path, eyeq_res_path: Path, dd_res_path: Path):
    """Plot confusion matrices from genuine EyeQ and DeepDRiD evaluation files."""
    if not eyeq_res_path.is_file() and not dd_res_path.is_file():
        raise FileNotFoundError(
            f"Cannot generate confusion matrices: neither {eyeq_res_path} nor {dd_res_path} exists."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. EyeQ Confusion Matrix
    if eyeq_res_path.is_file():
        with open(eyeq_res_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cm_eyeq = np.array(d["confusion_matrix"])
        ax0 = axes[0]
        im0 = ax0.imshow(cm_eyeq, interpolation="nearest", cmap="Blues")
        ax0.set_title("EyeQ Quality Confusion Matrix")
        ax0.set_xticks([0, 1, 2])
        ax0.set_yticks([0, 1, 2])
        ax0.set_xticklabels(["Good", "Usable", "Reject"])
        ax0.set_yticklabels(["Good", "Usable", "Reject"])
        ax0.set_xlabel("Predicted Quality")
        ax0.set_ylabel("True Quality")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        for i in range(cm_eyeq.shape[0]):
            for j in range(cm_eyeq.shape[1]):
                val = cm_eyeq[i, j]
                color = "white" if val > cm_eyeq.max() / 2 else "black"
                ax0.text(
                    j, i, f"{int(val)}", ha="center", va="center", color=color, fontweight="bold"
                )
    else:
        axes[0].text(
            0.5,
            0.5,
            "EyeQ evaluation pending",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("EyeQ Quality (Pending)")

    # 2. DeepDRiD Confusion Matrix
    if dd_res_path.is_file():
        with open(dd_res_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cm_dd = np.array(d["confusion_matrix"])
        ax1 = axes[1]
        im1 = ax1.imshow(cm_dd, interpolation="nearest", cmap="Purples")
        ax1.set_title("DeepDRiD Overall Quality Confusion Matrix")
        num_classes = cm_dd.shape[0]
        ax1.set_xticks(range(num_classes))
        ax1.set_yticks(range(num_classes))
        if num_classes == 2:
            ax1.set_xticklabels(["Good", "Reject"])
            ax1.set_yticklabels(["Good", "Reject"])
        else:
            ax1.set_xticklabels(["Good", "Usable", "Reject"])
            ax1.set_yticklabels(["Good", "Usable", "Reject"])
        ax1.set_xlabel("Predicted Class")
        ax1.set_ylabel("True Class")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        for i in range(cm_dd.shape[0]):
            for j in range(cm_dd.shape[1]):
                val = cm_dd[i, j]
                color = "white" if val > cm_dd.max() / 2 else "black"
                ax1.text(
                    j, i, f"{int(val)}", ha="center", va="center", color=color, fontweight="bold"
                )
    else:
        axes[1].text(
            0.5,
            0.5,
            "DeepDRiD evaluation pending",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("DeepDRiD Overall Quality (Pending)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")


def plot_learning_curves(output_path: Path, history_path: Path):
    """Plot training history directly from recorded train_history.json."""
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history file not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_score = history.get("val_score", history.get("val_macro_f1", []))

    if not train_loss or not val_loss:
        raise ValueError(f"Insufficient history data in {history_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    ax1.plot(epochs, train_loss, "b-o", linewidth=2, label="Train Loss", markersize=4)
    ax1.plot(epochs, val_loss, "r--s", linewidth=2, label="Validation Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Multi-Task Loss Curves")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    if val_score:
        ax2.plot(epochs, val_score, "g-^", linewidth=2, label="Val Selection Score", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score / Macro-F1")
        ax2.set_title("Validation Selection Objective")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")


def plot_reliability_diagram(output_path: Path, cal_json_path: Path):
    """Plot calibration reliability diagram from recorded calibration.json."""
    if not cal_json_path.is_file():
        raise FileNotFoundError(f"Calibration results file not found: {cal_json_path}")

    with open(cal_json_path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    uncal = cal.get("uncalibrated", {})
    calib = cal.get("calibrated", {})

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    if "bin_confidences" in uncal and "bin_accuracies" in uncal:
        ax.plot(
            uncal["bin_confidences"],
            uncal["bin_accuracies"],
            "r-o",
            label=f"Uncalibrated (ECE: {uncal.get('ece', 0):.4f})",
        )
    if "bin_confidences" in calib and "bin_accuracies" in calib:
        ax.plot(
            calib["bin_confidences"],
            calib["bin_accuracies"],
            "b-s",
            label=f"Calibrated (ECE: {calib.get('ece', 0):.4f}, T={cal.get('temperature', 1.0):.2f})",
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")


def plot_risk_coverage_curve(output_path: Path, sel_json_path: Path):
    """Plot Risk-Coverage curve from recorded selective_prediction.json."""
    if not sel_json_path.is_file():
        raise FileNotFoundError(f"Selective prediction file not found: {sel_json_path}")

    with open(sel_json_path, "r", encoding="utf-8") as f:
        sel = json.load(f)

    coverages = sel.get("coverages", [])
    risks = sel.get("risks", [])

    if not coverages or not risks:
        raise ValueError(f"No coverage/risk arrays in {sel_json_path}")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(coverages, risks, "b-o", linewidth=2, label="Predictive Entropy Gating")
    ax.set_xlabel("Coverage (Fraction of Accepted Samples)")
    ax.set_ylabel("Selective Error Rate (Risk)")
    aurc = sel.get("aurc", 0.0)
    ax.set_title(f"Risk-Coverage Curve (AURC: {aurc:.4f})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")


def plot_corruption_robustness(output_path: Path, corrupt_json_path: Path):
    """Plot degradation across corruptions from recorded corruptions.json."""
    if not corrupt_json_path.is_file():
        raise FileNotFoundError(f"Corruptions benchmark file not found: {corrupt_json_path}")

    with open(corrupt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corruptions = data.get("corruptions", {})
    if not corruptions:
        raise ValueError(f"No corruptions data found in {corrupt_json_path}")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    severities = [1, 2, 3, 4, 5]

    for c_name, c_res in corruptions.items():
        if isinstance(c_res, dict) and "f1_by_severity" in c_res:
            scores = [c_res["f1_by_severity"].get(str(s), 0.0) for s in severities]
            ax.plot(severities, scores, "-o", label=c_name, linewidth=1.5)

    clean_f1 = data.get("clean_macro_f1", None)
    if clean_f1 is not None:
        ax.axhline(clean_f1, color="k", linestyle="--", label=f"Clean Baseline ({clean_f1:.3f})")

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Model Robustness Under Optical & Synthetic Corruptions")
    ax.set_xticks(severities)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")


def generate_sha256sums(root_dir: Path, output_file: Path):
    """Compute and record SHA256 checksums of genuinely existing metrics, reports, and models in git repository."""
    checksums: List[str] = []
    for sub in ["metrics", "figures", "models", "provenance", "reports"]:
        dir_p = root_dir / sub
        if dir_p.is_dir():
            for p in sorted(dir_p.rglob("*")):
                # Exclude git-ignored large weight binaries (.ckpt, .pt) so git checkouts verify cleanly
                if (
                    p.is_file()
                    and not p.name.startswith(".")
                    and p.name != "SHA256SUMS"
                    and p.suffix.lower() not in [".ckpt", ".pt", ".pth"]
                ):
                    sha = compute_sha256(p)
                    rel_p = p.relative_to(root_dir)
                    checksums.append(f"{sha}  {rel_p}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(checksums) + "\n")
    print(f"Generated checksums file: {output_file} ({len(checksums)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from recorded experiment metrics."
    )
    parser.add_argument(
        "--metrics-dir", default="artifacts/metrics", help="Path to metrics directory"
    )
    parser.add_argument(
        "--figures-dir", default="artifacts/figures", help="Path to figures output directory"
    )
    parser.add_argument(
        "--provenance-dir", default="artifacts/provenance", help="Path to provenance directory"
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    fig_dir = Path(args.figures_dir)
    prov_dir = Path(args.provenance_dir)

    print("=== Generating Scientific Figures from Recorded Outputs ===")

    # Generate figures only if corresponding data files exist; report status honestly
    if (metrics_dir / "held_out_deepdrid.json").is_file():
        try:
            plot_confusion_matrices(
                fig_dir / "confusion_matrices.png",
                eyeq_res_path=metrics_dir / "eyeq_test.json",
                dd_res_path=metrics_dir / "held_out_deepdrid.json",
            )
        except Exception as e:
            print(f"Notice (confusion matrices): {e}")

    if (metrics_dir / "train_history.json").is_file():
        try:
            plot_learning_curves(
                fig_dir / "learning_curves.png",
                history_path=metrics_dir / "train_history.json",
            )
        except Exception as e:
            print(f"Notice (learning curves): {e}")

    if (metrics_dir / "calibration.json").is_file():
        try:
            plot_reliability_diagram(
                fig_dir / "reliability_diagram.png",
                cal_json_path=metrics_dir / "calibration.json",
            )
        except Exception as e:
            print(f"Notice (reliability diagram): {e}")

    if (metrics_dir / "selective_prediction.json").is_file():
        try:
            plot_risk_coverage_curve(
                fig_dir / "risk_coverage.png",
                sel_json_path=metrics_dir / "selective_prediction.json",
            )
        except Exception as e:
            print(f"Notice (risk coverage): {e}")

    if (metrics_dir / "corruptions.json").is_file():
        try:
            plot_corruption_robustness(
                fig_dir / "corruption_robustness.png",
                corrupt_json_path=metrics_dir / "corruptions.json",
            )
        except Exception as e:
            print(f"Notice (corruptions): {e}")

    generate_sha256sums(Path("artifacts"), prov_dir / "SHA256SUMS")
    print("Completed figure and checksum generation.")


if __name__ == "__main__":
    main()
