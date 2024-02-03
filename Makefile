MAIN_CODE = tarzan
CODE = $(MAIN_CODE)

.PHONY: test
test:
	pytest tests

.PHONY: lint
lint:
	black --config pyproject.toml --diff --check $(CODE)
	isort --settings-path pyproject.toml --diff --check-only $(CODE)
	flake8 --config .flake8 $(CODE)

.PHONY: check
check: lint test

.PHONY: format
format:
	black --config pyproject.toml $(CODE)
	isort --settings-path pyproject.toml $(CODE)
