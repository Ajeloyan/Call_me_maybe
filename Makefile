PYTHON = python3
MODULE = src

.PHONY: install run debug clean lint lint-strict

install:
	@uv sync

run:
	@uv run $(PYTHON) -m $(MODULE) $(ARGS)

debug:
	@uv run $(PYTHON) -m pdb -m $(MODULE) $(ARGS)

clean:
	@rm -rf */__pycache__
	@rm -rf */.mypy_cache
	@rm -rf .mypy_cache
	@rm -rf __pycache__

lint:
	@uv run flake8 .
	@uv run mypy --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs .

lint-strict:
	@uv run flake8 .
	@uv run mypy --strict .