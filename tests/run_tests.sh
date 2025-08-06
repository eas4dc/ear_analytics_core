#!/bin/bash
# Run tests using the virtual environment if available, otherwise use system python

if [ -f ".venv/bin/python" ]; then
  .venv/bin/python -m pytest -v test_*.py
else
  python -m pytest -v test_*.py
fi
