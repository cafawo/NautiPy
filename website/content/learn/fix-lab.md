# Fix Lab

The Fix Lab compares position-fixing situations without turning your browser
into a second solver. Every scenario is calculated ahead of time through
NautiPy’s public API. JavaScript only switches between the stored results and
draws their local schematic.

<div class="fix-lab" data-fix-lab data-fixture-url="../../assets/data/fix-lab.json">
  <p data-fix-lab-status role="status" aria-live="polite">
    Interactive scenarios load here when JavaScript is available.
  </p>
  <div data-fix-lab-controls></div>
  <div
    data-fix-lab-visual
    tabindex="0"
    role="group"
    aria-label="Scrollable Fix Lab schematic"
  ></div>
  <div data-fix-lab-summary></div>
  <div
    data-fix-lab-fallback
    tabindex="0"
    role="group"
    aria-label="Static Fix Lab diagram and explanation"
  >
    <img
      src="../../assets/images/range-geometry.svg"
      alt="Fallback schematic of two range constraints with ambiguous, tangent, and no-solution geometry."
    >
    <p>
      The static schematic remains the complete fallback. The interactive
      comparison appears here when its small local data file and JavaScript
      load successfully.
    </p>
  </div>
  <noscript>
    <p>
      JavaScript is disabled. You can still use the explanations, source
      example, and static range-geometry figure on this page.
    </p>
  </noscript>
</div>

The visual is a **local teaching schematic**. Its east/north axes, circles,
lines, and ellipses help compare geometry; they are not a nautical chart.
Fixture positions, predicted observations, residuals, and uncertainty are
calculated on WGS84.

## Eight experiments

### Two ranges: why two answers are normal

Two range constraints commonly cross twice. Both candidates satisfy the same
two measurements, so choosing the visually convenient one would be inventing
information. NautiPy reports ambiguity.

Watch for:

- two separated candidate boats;
- equally valid range constraints; and
- `AMBIGUOUS`, not a selected position.

### A third range: adding independent evidence

A third, well-placed range generally favors one of the two intersections. It
also creates one residual degree of freedom, so inconsistent data can begin to
show itself in fit metrics.

Watch for:

- one selected position;
- rank-two geometry; and
- residuals that remain small relative to the declared range uncertainties.

### Tangent ranges: one point can still be weak

Two range circles that merely touch have one mathematical candidate, but a
small measurement change can make them cross twice or not meet at all. The
candidate helper calls the point unique and warns about rank deficiency; the
general fixer calls the geometry degenerate.

Watch for the difference between candidate count and a stable two-dimensional
fix.

### Strong bearing geometry

Bearings toward separated references cross at a healthy angle. A small angular
change moves the intersection only modestly.

Watch for a compact uncertainty ellipse and a condition number nearer 1 than
in the weak case.

### Weak bearing geometry

Nearly parallel bearing constraints can intersect, yet leave one local
direction poorly determined. A small bearing error can move the fix a long
distance along that weak direction.

Watch for:

- an elongated ellipse;
- a large condition number or geometry warning; and
- the difference between “the optimizer stopped” and “the geometry is good.”

### Noisy mixed observations

Bearings and ranges can be fitted together only after each residual is divided
by its uncertainty. This scenario makes the natural units visible beside the
standardized residuals.

Watch how a metre residual and a degree residual become comparable only after
standardization.

### Uncertainty changes influence

An observation with a smaller declared standard deviation has more influence
on the standardized least-squares objective. This experiment changes
uncertainty while keeping the physical units explicit.

Watch the fitted position move toward the observation assigned the tighter
uncertainty. That weighting is meaningful only when the supplied uncertainty
model is defensible.

### Larger uncertainty, wider ellipse

The final scenario focuses on the 95% confidence ellipse. Its long axis points
toward the locally weak direction. It summarizes the supplied independent
standard deviations and the linearized model near the solution.

It is not a containment guarantee or safety boundary.

## Recreate the idea in Python

This complete example constructs exact ranges to a teaching position and adds
a third reference to resolve the usual two-range ambiguity:

```python
from nautipy import (
    Position,
    RangeObservation,
    distance,
    solve_fix,
    two_range_candidates,
)

boat_for_example = Position(50.12257, 8.66570)
references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)
ranges = tuple(
    RangeObservation(
        reference,
        distance(boat_for_example, reference),
        uncertainty=2.0,
    )
    for reference in references
)

two_range_result = two_range_candidates(ranges[0], ranges[1])
print(two_range_result.status, two_range_result.positions)

fix_result = solve_fix(ranges=ranges)
print(fix_result.status, fix_result.position)
print(fix_result.rank, fix_result.condition_number)
print(fix_result.uncertainty)
```

The fixture generator follows the same principle: use public NautiPy
operations, save ordinary numeric results, then let the browser present them.
The web page does not reproduce GeographicLib or the SciPy optimizer.

## Questions to ask as you compare

1. How many positions fit the observations?
2. Are both local directions constrained?
3. Are residuals small relative to the stated uncertainties?
4. Does the search domain limit the uniqueness claim?
5. Is the local uncertainty small enough for the intended purpose?
6. Which real-world effects are absent from the model?

Use the [result-reading guide](trusting-a-fix.md) for the meaning of every
diagnostic and [Finding the Boat](finding-the-boat.md) for the underlying
geometry.

## Limitations

- All scenarios are precomputed examples, not live measurements.
- Screen coordinates are local schematics and may exaggerate small
  differences for legibility.
- Bearings are true bearings at the unknown boat toward references.
- Range rings represent WGS84 surface distance; a drawn circle is not a claim
  that Earth is flat.
- The uncertainty model assumes independent Gaussian observation errors and
  no shared bias.

> **Navigation safety**
>
> The lab is for learning. It does not accept live navigation data and must
> not be treated as a chart, receiver, alarm, or certified fixing system.

## Learn more

- [Trilateration](https://en.wikipedia.org/wiki/Trilateration) introduces
  range-intersection geometry.
- [Resection](https://en.wikipedia.org/wiki/Resection_(orientation)) introduces
  position estimation from known references.
- [Confidence region](https://en.wikipedia.org/wiki/Confidence_region)
  provides background for the ellipse.
- NautiPy’s
  [position-fix behavior specification](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md)
  defines the calculations represented by the fixtures.
