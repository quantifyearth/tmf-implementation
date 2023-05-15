.PHONY: test
test:
				python3 -m pytest

.PHONY: type
type:
				python3 -m mypy -p methods
