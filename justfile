# pulsesuite — run `just` for all checks, `just --list` to browse

# run all checks
default: test lint fmt-check

# install all deps in editable mode (core + test + doc)
sync:
    uv sync --all-extras

# run the test suite (pass extra args: just test -k coulomb)
test *args:
    uv run pytest {{args}}

# lint with ruff
lint:
    uv run ruff check src/ tests/

# auto-fix lint issues
fix:
    uv run ruff check --fix src/ tests/

# format code
fmt:
    uv run ruff format src/ tests/

# check formatting without changes
fmt-check:
    uv run ruff format --check src/ tests/

# build sphinx docs
docs:
    uv run --group doc sphinx-build -b html docs/source docs/_build/html

# regenerate the lockfile after changing deps
lock:
    uv lock

# clean build artifacts and caches
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    rm -rf dist/ build/
