# NautiPy website

This directory contains the source for the educational site published at
<https://wbk.ing/NautiPy/>. It is intentionally excluded from NautiPy's wheel
and source distribution.

From the repository root, install the package and the pinned site tooling,
generate the verified Fix Lab data, and start a local preview:

```console
python -m pip install -e .
python -m pip install -r website/requirements.txt
python website/tools/generate_fix_lab.py
python -m unittest discover -s website/tests -v
python -m mkdocs serve --config-file website/mkdocs.yml
```

Run a production-equivalent build with:

```console
python -m mkdocs build --clean --strict --config-file website/mkdocs.yml
```
