"""Training runner script supporting config-driven execution, smoke tests, and isolated fixture mode."""

import argparse

from retinaguard.training.train import run_training_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Train RetinaGuard models with fail-fast validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_multitask.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test for CI/CD")
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Run training using isolated test fixtures in tests/fixtures/",
    )
    args = parser.parse_args()

    print(
        f"=== Starting Training with Config: {args.config} (Smoke: {args.smoke_test}, Fixture: {args.fixture_mode}) ==="
    )
    results = run_training_experiment(
        args.config, smoke_test=args.smoke_test, fixture_mode=args.fixture_mode
    )
    print(f"Training successfully completed! Best Val Macro-F1: {results['best_val_macro_f1']:.4f}")
    print(f"Checkpoint saved at: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
