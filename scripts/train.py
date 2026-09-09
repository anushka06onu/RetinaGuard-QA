"""Training runner script supporting config-driven execution and smoke tests."""

import argparse

from src.retinaguard.training.train import run_training_experiment


def main():
    parser = argparse.ArgumentParser(description="Train RetinaGuard models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_multitask.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test for CI/CD")
    args = parser.parse_args()

    print(f"=== Starting Training with Config: {args.config} (Smoke test: {args.smoke_test}) ===")
    results = run_training_experiment(args.config, smoke_test=args.smoke_test)
    print(f"Training successfully completed! Best Val Macro-F1: {results['best_val_macro_f1']:.4f}")
    print(f"Checkpoint saved at: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
