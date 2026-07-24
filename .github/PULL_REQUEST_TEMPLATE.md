## Summary

Describe the user problem and the observable behavior this pull request adds,
changes, or fixes.

## Examples or references

Include a minimal example, independent numerical reference, or explanation of
why no example is needed.

## Verification

List the exact commands run and their results.

```text
python -m pip check
python -m unittest discover -s tests -v
```

## Checklist

- [ ] The change is focused and fits NautiPy's documented product scope.
- [ ] Public behavior and relevant error cases have meaningful tests.
- [ ] User-facing behavior, examples, and the changelog are updated where
      needed.
- [ ] Public API and runtime dependency effects are explicit.
- [ ] Numerical units, assumptions, tolerances, and limitations are documented.
- [ ] Packaging or import changes were tested from the built wheel and sdist.
- [ ] No credentials or private position data are included.
