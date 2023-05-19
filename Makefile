.PHONY: test
test:
				python3 -m pytest

.PHONY: type
type:
				python -m mypy methods

.PHONY: lint
lint:
				python -m pylint main.py
				python -m pylint methods
