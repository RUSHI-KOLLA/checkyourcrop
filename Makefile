.PHONY: help install test lint build run evaluate register deploy smoke clean

help:
	@echo "CheckYourCrop MLOps Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install dependencies (production)"
	@echo "  install-dev  Install dependencies (development)"
	@echo "  test         Run unit tests"
	@echo "  lint         Run ruff linter"
	@echo "  build        Build Docker image"
	@echo "  run          Run API server locally"
	@echo "  pipeline     Run full pipeline (lint,test,build,evaluate,register,deploy,smoke)"
	@echo "  evaluate     Run evaluation gate"
	@echo "  register     Register model to HF Hub"
	@echo "  deploy       Deploy to HF Spaces"
	@echo "  smoke        Smoke test deployed service"
	@echo "  clean        Clean build artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	ruff check app tests scripts

build:
	docker build -t checkyourcrop:test .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

pipeline:
	python scripts/pipeline.py --stage all

evaluate:
	python scripts/evaluate.py --baseline 0.0 --data eval/golden/data.yaml --weights models/best.pt

register:
	HF_TOKEN=$$HF_TOKEN HF_MODEL_REPO=$$HF_MODEL_REPO python scripts/pipeline.py --stage register

deploy:
	HF_TOKEN=$$HF_TOKEN HF_SPACE=$$HF_SPACE python scripts/pipeline.py --stage deploy

smoke:
	SERVICE_URL=$$SERVICE_URL python scripts/pipeline.py --stage smoke

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov build dist *.egg-info
	rm -rf outputs runs logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true