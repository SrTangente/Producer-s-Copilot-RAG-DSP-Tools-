.PHONY: install dev lint test run docker

install:
	pip install --upgrade pip && pip install -r requirements-dev.txt

lint:
	pylint --disable=R,C app

test:
	python -m pytest -q

run:
	uvicorn app.main:app --reload --port 8000

docker:
	docker build -t producers-copilot . && docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY producers-copilot
