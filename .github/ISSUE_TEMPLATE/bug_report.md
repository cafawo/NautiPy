---
name: Bug report
about: Report reproducible incorrect behavior in NautiPy
title: ""
labels: ""
assignees: ""
---

## Summary

Describe what went wrong and which NautiPy workflow is affected.

## Minimal reproduction

Provide the smallest runnable Python example or command-line invocation that
shows the problem. Use synthetic coordinates if the real positions are
private.

```python
# Your example here
```

## Expected behavior

Describe the expected result. For numerical reports, include the units,
tolerance, and an independent reference or derivation when available.

## Actual behavior

Include the complete exception and traceback or the unexpected result.

```text
Paste output here
```

## Environment

- NautiPy version or commit:
- Python version:
- Operating system and architecture:
- GeographicLib version:
- NumPy version:
- SciPy version:
- Installation method:

`python -m pip show nautipy geographiclib numpy scipy` can provide most package
versions.

## Coordinate and numerical context

When relevant, include the input format, coordinate order, units, observation
uncertainties, search bounds, and any assumptions used to interpret the data.

## Additional context

Add anything else that helps reproduce or diagnose the problem.

Do not file suspected vulnerabilities, credentials, or private position data
here. Follow the
[security policy](https://github.com/cafawo/NautiPy/security/policy) for
private reporting.
