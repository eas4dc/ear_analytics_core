# ear_analytics_core
Core library for EAR data analytics.

## Installation

Since version 5.0.0, `ear_analytics_core` was uploaded to PyPI and can be installed via pip:
```bash
pip install ear_analytics_core
```

Alternatively, you can install it from source by cloning the repository and running:

```bash
pip install -U pip
pip install build setuptools wheel
python -m build
pip install .
```

> You can change the destination path by exporting the variable [`PYTHONUSERBASE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUSERBASE).
> Tool's developers may want to use `pip install -e .` to install the package in editable mode, so there is no need to reinstall every time you want to test a new feature.
