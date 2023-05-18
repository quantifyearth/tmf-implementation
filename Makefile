.PHONY: test
test:
				python3 -m pytest

.PHONY: type
type:
				python -m mypy methods
				python -m mypy arkdir

.PHONY: lint
lint:
				python -m pylint arkdir
