.PHONY: dev
dev:
	uv sync
	uv run prek install

.PHONY: pre
pre: format lint test

.PHONY: format
format:
	ruff format

.PHONY: lint
lint:
	ruff check --fix

.PHONY: check
check:
	ruff format --check && ruff check

.PHONY: test
test:
	uv run --frozen pytest

.PHONY: build
build:
	uv build --no-create-gitignore --no-sources

.PHONY: clean
clean:
	rm -rf dist/ .pytest_cache/
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	ruff clean
