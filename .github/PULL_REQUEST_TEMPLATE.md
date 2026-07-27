## Summary

Describe the user problem and the observable behavior this pull request adds,
changes, or fixes.

## Examples or references

Include a minimal example, independent numerical reference, or explanation of
why no example is needed.

## Documentation impact

List the exact behavior specification and educational website pages updated.
If there are no documentation edits, write `Documentation impact: none` and
explain why the change has no externally observable effect.

## Verification

List the exact commands run and their results.

```text
python -m pip check
python -m unittest discover -s tests -v
```

## Checklist

- [ ] The change is focused and fits NautiPy's documented product scope.
- [ ] Public behavior and relevant error cases have meaningful tests.
- [ ] Public functionality changes update both the authoritative behavior
      specification and the affected educational website material.
- [ ] Examples, visuals, Fix Lab fixtures, and the changelog are updated where
      their meaning changes.
- [ ] The documentation-impact section names the updated files or gives a
      concrete no-impact explanation.
- [ ] Website tests and the strict MkDocs build pass when public functionality
      or website content changes.
- [ ] Public API and runtime dependency effects are explicit.
- [ ] Numerical units, assumptions, tolerances, and limitations are documented.
- [ ] Packaging or import changes were tested from the built wheel and sdist.
- [ ] No credentials or private position data are included.
