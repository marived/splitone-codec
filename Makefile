test:
	pytest -q tests/

lint:
	ruff check splitone tests
	black --check splitone tests

fmt:
	black splitone tests
	ruff check --fix splitone tests
