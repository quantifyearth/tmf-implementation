.PHONY: test
test:
				python3 -m pytest -vv

.PHONY: type
type:
				python -m mypy methods

.PHONY: lint
lint:
				python -m pylint methods
