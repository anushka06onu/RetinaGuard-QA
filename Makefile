.PHONY: install audit-data split-data verify-splits smoke-train test train evaluate calibrate benchmark export campaign ablations verify-artifacts docker-smoke api web docker lint format

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	cd web && npm ci

audit-data:
	PYTHONPATH=src $(PYTHON) scripts/audit_dataset.py

split-data:
	PYTHONPATH=src $(PYTHON) scripts/create_splits.py

verify-splits:
	$(PYTHON) scripts/verify_splits.py

verify-artifacts:
	$(PYTHON) scripts/verify_artifacts.py

smoke-train:
	PYTHONPATH=src $(PYTHON) scripts/train.py --config configs/train_eyeq.yaml --smoke-test

train:
	PYTHONPATH=src $(PYTHON) scripts/train.py --config configs/train_eyeq.yaml
	PYTHONPATH=src $(PYTHON) scripts/train.py --config configs/train_multitask.yaml

campaign:
	PYTHONPATH=src $(PYTHON) scripts/run_campaign.py --config configs/train_multitask.yaml --seeds 42 43 44

ablations:
	PYTHONPATH=src $(PYTHON) scripts/run_ablations.py --config configs/train_multitask.yaml

evaluate:
	PYTHONPATH=src $(PYTHON) scripts/evaluate.py --checkpoint artifacts/models/best.ckpt

calibrate:
	PYTHONPATH=src $(PYTHON) scripts/calibrate.py --checkpoint artifacts/models/best.ckpt

benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_corruptions.py --checkpoint artifacts/models/best.ckpt
	PYTHONPATH=src $(PYTHON) scripts/benchmark_ood.py --checkpoint artifacts/models/best.ckpt
	PYTHONPATH=src $(PYTHON) scripts/benchmark_inference.py --model artifacts/models/model.onnx

export:
	PYTHONPATH=src $(PYTHON) scripts/export_onnx.py --checkpoint artifacts/models/best.ckpt

test:
	PYTHONPATH=src pytest tests/ api/tests/ -v

lint:
	ruff check .
	black --check .
	mypy src/

format:
	black .
	ruff check --fix .

api:
	$(PYTHON) -m uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm run dev

docker:
	docker compose up --build

docker-smoke:
	docker compose build
