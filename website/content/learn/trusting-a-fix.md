# Can You Trust the Fix?

A solver can stop successfully and still leave the underlying navigation
question unanswered. Perhaps two positions fit almost equally well. Perhaps
the observations constrain east–west motion but barely constrain north–south
motion. Perhaps the measurements disagree with their claimed uncertainties.

NautiPy’s `FixResult` separates four questions:

1. **Outcome:** did the search find one acceptable in-domain basin?
2. **Geometry:** do the observations constrain both local dimensions?
3. **Fit:** do predictions agree with observations at the declared noise
   scale?
4. **Uncertainty:** how does local measurement uncertainty map into position?

## Start with status, not the coordinate

`FixStatus.CONVERGED` is the only successful status.

| Status | Meaning |
| --- | --- |
| `CONVERGED` | One in-domain position converged with rank-two local geometry. |
| `AMBIGUOUS` | Multiple materially distinct, statistically comparable positions fit. |
| `DEGENERATE` | The observations do not stably constrain both local dimensions. |
| `NO_SOLUTION` | No valid solution exists in the declared search domain. |
| `NON_CONVERGED` | The numerical search ended without a valid converged basin. |

Every unsuccessful result has `position=None`. An ambiguous result instead
exposes its deterministic `competing_positions`; NautiPy does not choose one
because its numerical cost is microscopically lower.

```python
result = solve_fix(ranges=ranges)

if not result.success:
    print(result.status, result.message)
    print(result.competing_positions)
```

## Residuals put the observations on trial

At a trial position, NautiPy predicts every supplied observation.

- Bearing residual: wrapped `predicted − observed` in `[-180°, 180°)`.
- Range residual: `predicted − observed` in metres.
- Standardized residual: natural residual divided by the observation’s
  standard deviation.

In compact notation,

```text
standardized residual = (prediction − observation) / uncertainty
```

A positive range residual means the predicted range is too long. A positive
bearing residual means the predicted direction is clockwise from the observed
direction.

```python
for item in result.residuals:
    print(
        item.observation,
        item.predicted,
        item.residual,
        item.standardized_residual,
    )
```

`objective` is the sum of squared standardized residuals. `rms` is their
dimensionless root-mean-square value; `bearing_rms` and `range_rms` retain
natural units where applicable. An RMS above 2 produces a warning because
residuals are large relative to the uncertainties supplied.

This interpretation depends on the uncertainty model being reasonable. Tiny
declared uncertainties can make modest residuals look severe; inflated
uncertainties can hide a poor physical fit.

## Rank and condition number describe geometry

Near the solution, a Jacobian records how standardized predictions change
with small east and north movements.

- **Rank 2** means both local directions are constrained.
- Missing rank, or an extremely ill-conditioned system, is degenerate.
- The **condition number** compares the strong and weak local directions. Near
  1 is balanced; a large value means errors can be greatly amplified in one
  direction.

NautiPy warns above a condition number of 1,000 and classifies geometry as
degenerate when rank is missing or the condition number exceeds 1,000,000.
Those thresholds diagnose the local numerical model; they do not replace
domain knowledge.

Nearly parallel bearings and nearly tangent range circles are familiar
examples of weak geometry.

## Degrees of freedom and reduced chi-square

The fixer estimates two local coordinates. With *n* scalar observations,
the reported degrees of freedom are:

```text
degrees of freedom = n − 2
```

When this number is positive, `reduced_chi_square` is:

```text
objective / degrees of freedom
```

It helps compare the total standardized mismatch with the assumed independent
Gaussian errors. It is absent for a just-determined two-observation problem,
where no residual degrees of freedom remain to assess fit.

Do not read one realization as a universal pass/fail test. Correlation, bias,
non-Gaussian errors, and a wrong physical model can all invalidate the simple
statistical interpretation.

## The local 95% uncertainty ellipse

For a unique, converged, full-rank result, NautiPy linearizes the observation
model around the selected position.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![A local east-north confidence ellipse around a position fix, with semi-major
and semi-minor axes and the major-axis true
bearing.](../assets/images/uncertainty-ellipse.svg)

</div>

`FixUncertainty` reports:

- a 2 × 2 east/north covariance matrix in square metres;
- east and north standard deviations in metres;
- their correlation;
- semi-major and semi-minor axes of a local 95% ellipse in metres; and
- the major-axis true bearing in `[0°, 180°)`, unless the ellipse is isotropic.

```python
if result.uncertainty is not None:
    uncertainty = result.uncertainty
    print(uncertainty.east_standard_deviation)
    print(uncertainty.north_standard_deviation)
    print(uncertainty.semi_major_95)
    print(uncertainty.semi_minor_95)
    print(uncertainty.major_axis_bearing)
```

The covariance is `C = (JᵀJ)⁻¹` under the declared absolute
observation uncertainties. NautiPy does not rescale it by the fitted residual
variance.

The ellipse is a **local, linearized confidence description**, not a safety
boundary. It is withheld for ambiguous, non-converged, rank-deficient, or
numerically invalid geometry. Large uncertainty relative to the search domain
also produces a warning.

## Remember the search domain

The fixer searches a closed WGS84-distance disk. A result near its boundary
warns that the chosen domain matters. A solution outside the disk is not
projected onto the edge, and a uniqueness claim never extends to the entire
globe.

When choosing a domain:

- center it from defensible prior knowledge or accept the deterministic
  reference-derived default;
- make it large enough for the physical problem, but not a substitute for a
  global solver; and
- record it alongside the result because it limits the claim.

## A compact review checklist

```python
print("status:", result.status)
print("message:", result.message)
print("warnings:", result.warnings)
print("rank:", result.rank)
print("condition:", result.condition_number)
print("standardized RMS:", result.rms)
print("reduced chi-square:", result.reduced_chi_square)
print("competing positions:", result.competing_positions)
print("uncertainty:", result.uncertainty)
```

Ask whether the references and observations are independent, correctly timed,
in the documented units, and free of known common bias. Diagnostics cannot
detect every bad assumption.

For exact public thresholds and result invariants, see the
[position-fix behavior specification](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md).

> **Navigation safety**
>
> Neither convergence nor a 95% ellipse certifies a safe position. Maintain
> independent checks appropriate to the vessel, environment, and consequences.

## Learn more

- [Least squares](https://en.wikipedia.org/wiki/Least_squares) introduces the
  fitting principle.
- [Condition number](https://en.wikipedia.org/wiki/Condition_number) explains
  sensitivity to input errors.
- [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) and
  [confidence region](https://en.wikipedia.org/wiki/Confidence_region)
  introduce the uncertainty language.
- [SciPy’s bounded nonlinear least-squares reference](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.least_squares.html)
  documents the numerical optimizer used by NautiPy.
- NIST’s
  [Uncertainty of Measurement resources](https://physics.nist.gov/cuu/Uncertainty/index.html)
  provide primary guidance on stating measurement uncertainty.

Try the [Fix Lab](fix-lab.md).
