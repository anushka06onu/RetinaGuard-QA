.PHONY: install audit-data split-data smoke-train test train evaluate calibrate benchmark export api web docker lint format

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	cd web && npm install

audit-data:
	PYTHONPATH=. $(PYTHON) scripts/audit_dataset.py

split-data:
	PYTHONPATH=. $(PYTHON) scripts/create_splits.py

smoke-train:
	PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/train_eyeq.yaml --smoke-test

train:
	PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/train_eyeq.yaml
	PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/train_multitask.yaml

evaluate:
	PYTHONPATH=. $(PYTHON) scripts/evaluate.py --checkpoint artifacts/models/best.ckpt

calibrate:
	PYTHONPATH=. $(PYTHON) scripts/calibrate.py --checkpoint artifacts/models/best.ckpt

benchmark:
	PYTHONPATH=. $(PYTHON) scripts/benchmark_corruptions.py --checkpoint artifacts/models/best.ckpt
	PYTHONPATH=. $(PYTHON) scripts/benchmark_inference.py --model artifacts/models/model.onnx

export:
	PYTHONPATH=. $(PYTHON) scripts/export_onnx.py --checkpoint artifacts/models/best.ckpt

test:
	PYTHONPATH=. pytest tests/ api/tests/ -v

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
