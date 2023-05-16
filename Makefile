.PHONY: test
test:
				python3 -m pytest

.PHONY: type
type:
				python -m mypy methods
