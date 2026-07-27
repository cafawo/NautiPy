"""Generate deterministic, precomputed data for the educational Fix Lab.

The browser only draws these results.  Every geodesic, candidate, fix, and
uncertainty value is produced here through NautiPy's documented top-level API.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from nautipy import (
    BearingObservation,
    CandidateStatus,
    FixStatus,
    Position,
    RangeObservation,
    destination,
    distance,
    initial_bearing,
    solve_fix,
    two_bearing_candidates,
    two_range_candidates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "website"
    / "content"
    / "assets"
    / "data"
    / "fix-lab.json"
)


def _rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Fix Lab fixture check failed: {message}")


def _local_coordinates(origin: Position, position: Position) -> tuple[float, float]:
    if position == origin:
        return (0.0, 0.0)
    separation = distance(origin, position)
    bearing = initial_bearing(origin, position)
    if bearing is None:
        return (0.0, 0.0)
    angle = math.radians(bearing)
    return (
        separation * math.sin(angle),
        separation * math.cos(angle),
    )


def _point(
    identifier: str,
    label: str,
    kind: str,
    position: Position,
    origin: Position,
) -> dict[str, Any]:
    east, north = _local_coordinates(origin, position)
    return {
        "id": identifier,
        "label": label,
        "kind": kind,
        "latitude": _rounded(position.latitude, 8),
        "longitude": _rounded(position.longitude, 8),
        "east_m": _rounded(east, 3),
        "north_m": _rounded(north, 3),
    }


def _reference_points(
    references: Iterable[Position],
    origin: Position,
) -> list[dict[str, Any]]:
    return [
        _point(f"reference-{index}", f"Reference {index}", "reference", item, origin)
        for index, item in enumerate(references, start=1)
    ]


def _observations(
    bearings: Iterable[BearingObservation] = (),
    ranges: Iterable[RangeObservation] = (),
    *,
    origin_id: str = "truth",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_observations = (*tuple(bearings), *tuple(ranges))
    references: list[Position] = []
    for observation in all_observations:
        if observation.reference not in references:
            references.append(observation.reference)
        reference_id = f"reference-{references.index(observation.reference) + 1}"
        if isinstance(observation, BearingObservation):
            rows.append(
                {
                    "kind": "bearing",
                    "reference_id": reference_id,
                    "origin_id": origin_id,
                    "value": _rounded(observation.bearing, 4),
                    "unit": "degrees true",
                    "uncertainty": _rounded(observation.uncertainty, 4),
                }
            )
        else:
            rows.append(
                {
                    "kind": "range",
                    "reference_id": reference_id,
                    "value": _rounded(observation.distance, 3),
                    "unit": "metres",
                    "uncertainty": _rounded(observation.uncertainty, 3),
                }
            )
    return rows


def _uncertainty_payload(
    result: Any,
    *,
    identifier: str,
    label: str,
    display_scale: float = 1.0,
) -> dict[str, Any]:
    uncertainty = result.uncertainty
    _require(uncertainty is not None, f"{identifier} has no uncertainty")
    return {
        "id": identifier,
        "label": label,
        "center_id": "fix",
        "semi_major_95_m": _rounded(uncertainty.semi_major_95, 3),
        "semi_minor_95_m": _rounded(uncertainty.semi_minor_95, 3),
        "major_axis_bearing": _rounded(uncertainty.major_axis_bearing, 3),
        "display_scale": display_scale,
        "note": "Local linearized 95% confidence ellipse; not a safety bound.",
    }


def _diagnostics(result: Any) -> dict[str, Any]:
    residuals = []
    counts = {"bearing": 0, "range": 0}
    for item in result.residuals:
        is_bearing = isinstance(item.observation, BearingObservation)
        kind = "bearing" if is_bearing else "range"
        counts[kind] += 1
        residuals.append(
            {
                "label": f"{kind.title()} {counts[kind]}",
                "kind": kind,
                "natural_residual": _rounded(
                    item.residual,
                    6 if is_bearing else 3,
                ),
                "natural_unit": "degrees" if is_bearing else "metres",
                "standardized_residual": _rounded(
                    item.standardized_residual,
                    4,
                ),
            }
        )
    return {
        "status": result.status.value,
        "message": result.message,
        "rms": _rounded(result.rms, 4),
        "bearing_rms_degrees": _rounded(result.bearing_rms, 4),
        "range_rms_m": _rounded(result.range_rms, 3),
        "rank": result.rank,
        "condition_number": _rounded(result.condition_number, 3),
        "warnings": list(result.warnings),
        "residuals": residuals,
    }


def _two_range_scenario() -> dict[str, Any]:
    truth = Position(50.127198, 8.665562)
    references = (
        Position(50.116135, 8.670277),
        Position(50.110347, 8.659873),
    )
    ranges = tuple(
        RangeObservation(reference, distance(truth, reference), 2.0)
        for reference in references
    )
    candidates = two_range_candidates(*ranges)
    _require(candidates.status is CandidateStatus.AMBIGUOUS, "two ranges changed")
    _require(len(candidates.positions) == 2, "two ranges need two candidates")

    positions = [_point("truth", "Actual boat", "truth", truth, truth)]
    positions.extend(
        _point(
            f"candidate-{index}",
            f"Candidate {index}",
            "candidate",
            position,
            truth,
        )
        for index, position in enumerate(candidates.positions, start=1)
    )
    return {
        "id": "two-ranges",
        "title": "Two ranges: two possible boats",
        "lesson": (
            "Two WGS84 range circles commonly intersect twice, so NautiPy "
            "reports both candidates instead of guessing."
        ),
        "schematic_note": (
            "The screen is a local east/north schematic; the fixture values "
            "were calculated on WGS84."
        ),
        "references": _reference_points(references, truth),
        "observations": _observations(ranges=ranges),
        "positions": positions,
        "ellipses": [],
        "diagnostics": {
            "status": candidates.status.value,
            "message": candidates.message,
            "candidate_count": len(candidates.positions),
            "warnings": list(candidates.warnings),
        },
    }


def _third_range_scenario() -> dict[str, Any]:
    truth = Position(50.127198, 8.665562)
    references = (
        Position(50.116135, 8.670277),
        Position(50.110347, 8.659873),
        Position(50.112836, 8.666753),
    )
    ranges = tuple(
        RangeObservation(reference, distance(truth, reference), 2.0)
        for reference in references
    )
    result = solve_fix(
        ranges=ranges,
        search_center=truth,
        search_radius=10_000,
    )
    _require(result.status is FixStatus.CONVERGED, "third range did not converge")
    _require(distance(result.position, truth) < 0.01, "Frankfurt fix moved")

    return {
        "id": "third-range",
        "title": "A third range resolves the ambiguity",
        "lesson": (
            "The third observation agrees with only one of the two-range "
            "candidates, producing a unique diagnosed fix."
        ),
        "schematic_note": (
            "The screen is a local east/north schematic; the fixture values "
            "were calculated on WGS84. The compact ellipse is drawn at 20× "
            "scale for legibility."
        ),
        "references": _reference_points(references, truth),
        "observations": _observations(ranges=ranges),
        "positions": [
            _point("truth", "Actual boat", "truth", truth, truth),
            _point("fix", "NautiPy fix", "fix", result.position, truth),
        ],
        "ellipses": [
            _uncertainty_payload(
                result,
                identifier="fix-uncertainty",
                label="95% local uncertainty",
                display_scale=20.0,
            )
        ],
        "diagnostics": _diagnostics(result),
    }


def _tangent_range_scenario() -> dict[str, Any]:
    references = (Position(0, -0.01), Position(0, 0.01))
    center = Position(0, 0)
    tangent_distance = distance(*references) / 2
    ranges = tuple(
        RangeObservation(reference, tangent_distance, 1.0)
        for reference in references
    )
    candidates = two_range_candidates(*ranges)
    result = solve_fix(
        ranges=ranges,
        search_center=center,
        search_radius=5_000,
    )
    _require(candidates.status is CandidateStatus.UNIQUE, "tangent candidate changed")
    _require(result.status is FixStatus.DEGENERATE, "tangent fix must be degenerate")

    return {
        "id": "tangent-ranges",
        "title": "Touching ranges: one point, weak geometry",
        "lesson": (
            "Tangent circles touch at one mathematical point, but they do not "
            "constrain two stable directions, so the full fix is degenerate."
        ),
        "schematic_note": (
            "The screen is a local east/north schematic; the fixture values "
            "were calculated on WGS84."
        ),
        "references": _reference_points(references, center),
        "observations": _observations(ranges=ranges, origin_id="candidate-1"),
        "positions": [
            _point(
                "candidate-1",
                "Tangent candidate",
                "candidate",
                candidates.positions[0],
                center,
            )
        ],
        "ellipses": [],
        "diagnostics": {
            **_diagnostics(result),
            "candidate_status": candidates.status.value,
            "candidate_message": candidates.message,
        },
    }


def _bearing_scenario(*, weak: bool) -> dict[str, Any]:
    truth = Position(50.127198, 8.665562)
    angular_separation = 0.1 if weak else 90.0
    references = (
        destination(truth, bearing=0, distance=2_000),
        destination(truth, bearing=angular_separation, distance=2_000),
    )
    bearings = tuple(
        BearingObservation(
            reference,
            initial_bearing(truth, reference),
            0.1,
        )
        for reference in references
    )
    candidates = two_bearing_candidates(
        *bearings,
        search_center=truth,
        search_radius=5_000,
    )
    result = solve_fix(
        bearings=bearings,
        search_center=truth,
        search_radius=5_000,
    )
    _require(candidates.status is CandidateStatus.UNIQUE, "bearing candidate changed")
    _require(result.status is FixStatus.CONVERGED, "bearing fix did not converge")
    _require(result.condition_number is not None, "bearing condition unavailable")
    if weak:
        _require(result.condition_number > 1_000, "weak bearing geometry is not weak")
        _require(
            any("weak" in warning for warning in result.warnings),
            "weak bearing warning missing",
        )
    else:
        _require(result.condition_number < 10, "strong bearing geometry weakened")

    adjective = "Weak" if weak else "Strong"
    return {
        "id": "weak-bearings" if weak else "strong-bearings",
        "title": f"{adjective} bearing geometry",
        "lesson": (
            "Nearly parallel sight lines amplify small angular errors and "
            "stretch the uncertainty ellipse."
            if weak
            else "Sight lines crossing near a right angle constrain both local axes."
        ),
        "schematic_note": (
            "Bearing arrows start at the boat and point toward each reference. "
            "The screen is schematic; calculations use true WGS84 bearings. "
            + (
                "The extreme ellipse shape is shown at its natural scale."
                if weak
                else "The compact ellipse is drawn at 10× scale for legibility."
            )
        ),
        "references": _reference_points(references, truth),
        "observations": _observations(bearings=bearings),
        "positions": [
            _point("truth", "Actual boat", "truth", truth, truth),
            _point("fix", "NautiPy fix", "fix", result.position, truth),
        ],
        "ellipses": [
            _uncertainty_payload(
                result,
                identifier="fix-uncertainty",
                label="95% local uncertainty",
                display_scale=1.0 if weak else 10.0,
            )
        ],
        "diagnostics": {
            **_diagnostics(result),
            "candidate_status": candidates.status.value,
        },
    }


def _noisy_mixed_scenario() -> dict[str, Any]:
    truth = Position(50.127198, 8.665562)
    references = tuple(
        destination(truth, bearing=bearing, distance=separation)
        for bearing, separation in ((20, 1_800), (140, 1_600), (260, 1_900))
    )
    bearings = (
        BearingObservation(
            references[0],
            initial_bearing(truth, references[0]) + 0.08,
            0.1,
        ),
        BearingObservation(
            references[1],
            initial_bearing(truth, references[1]) - 0.05,
            0.1,
        ),
    )
    ranges = (
        RangeObservation(references[1], distance(truth, references[1]) + 7, 5),
        RangeObservation(references[2], distance(truth, references[2]) - 5, 5),
    )
    result = solve_fix(
        bearings=bearings,
        ranges=ranges,
        search_center=truth,
        search_radius=10_000,
    )
    _require(result.status is FixStatus.CONVERGED, "noisy mixed fix did not converge")
    _require(0 < distance(result.position, truth) < 10, "noisy fix shift changed")

    return {
        "id": "noisy-mixed",
        "title": "Noisy bearings and ranges",
        "lesson": (
            "Real observations rarely meet at one exact point. Standardized "
            "least squares balances their residuals using stated uncertainties."
        ),
        "schematic_note": (
            "Bearing arrows start at the actual boat and point toward references. "
            "The screen is schematic; predictions and residuals use WGS84. "
            "The compact ellipse is drawn at 20× scale for legibility."
        ),
        "references": _reference_points(references, truth),
        "observations": _observations(bearings, ranges),
        "positions": [
            _point("truth", "Actual boat", "truth", truth, truth),
            _point("fix", "Estimated fix", "fix", result.position, truth),
        ],
        "ellipses": [
            _uncertainty_payload(
                result,
                identifier="fix-uncertainty",
                label="95% local uncertainty",
                display_scale=20.0,
            )
        ],
        "diagnostics": _diagnostics(result),
    }


def _weighting_scenario() -> dict[str, Any]:
    truth = Position(0.01, 0.01)
    references = (
        Position(0, 0),
        Position(0, 0.02),
        Position(0.02, 0),
        Position(0.02, 0.02),
    )
    exact_ranges = tuple(distance(truth, reference) for reference in references)

    def solve_with(biased_uncertainty: float) -> tuple[Any, tuple[RangeObservation, ...]]:
        ranges = (
            RangeObservation(
                references[0],
                exact_ranges[0] + 200,
                biased_uncertainty,
            ),
            *tuple(
                RangeObservation(reference, measured, 10)
                for reference, measured in zip(references[1:], exact_ranges[1:])
            ),
        )
        return (
            solve_fix(
                ranges=ranges,
                search_center=truth,
                search_radius=10_000,
            ),
            ranges,
        )

    high_weight, high_weight_ranges = solve_with(1)
    low_weight, _ = solve_with(1_000)
    high_shift = distance(high_weight.position, truth)
    low_shift = distance(low_weight.position, truth)
    _require(high_weight.status is FixStatus.CONVERGED, "high-weight fix failed")
    _require(low_weight.status is FixStatus.CONVERGED, "low-weight fix failed")
    _require(high_shift > low_shift * 100, "uncertainty no longer changes weighting")

    return {
        "id": "uncertainty-weighting",
        "title": "Uncertainty controls influence",
        "lesson": (
            "The same biased range pulls hard when declared precise and barely "
            "moves the fix when declared uncertain. Uncertainty is not decoration."
        ),
        "schematic_note": (
            "Both fixes use the same WGS84 observations; only the biased "
            "observation's standard deviation changes."
        ),
        "references": _reference_points(references, truth),
        "observations": _observations(ranges=high_weight_ranges),
        "positions": [
            _point("truth", "Actual boat", "truth", truth, truth),
            _point(
                "high-weight-fix",
                "Biased range: σ = 1 m",
                "comparison",
                high_weight.position,
                truth,
            ),
            _point(
                "low-weight-fix",
                "Biased range: σ = 1,000 m",
                "fix",
                low_weight.position,
                truth,
            ),
        ],
        "ellipses": [],
        "diagnostics": {
            "status": high_weight.status.value,
            "message": "Two converged fits compared",
            "high_weight_shift_m": _rounded(high_shift, 3),
            "low_weight_shift_m": _rounded(low_shift, 3),
            "warnings": list(high_weight.warnings),
        },
    }


def _uncertainty_scale_scenario() -> dict[str, Any]:
    truth = Position(0.011, 0.013)
    references = (
        Position(0, 0),
        Position(0, 0.03),
        Position(0.03, 0),
        Position(0.025, 0.028),
    )
    factor = 8.5

    def solve_with(scale: float) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        bearings = tuple(
            BearingObservation(
                reference,
                initial_bearing(truth, reference),
                0.2 * scale,
            )
            for reference in references[:2]
        )
        ranges = tuple(
            RangeObservation(
                reference,
                distance(truth, reference),
                3.0 * scale,
            )
            for reference in references[2:]
        )
        return (
            solve_fix(
                bearings=bearings,
                ranges=ranges,
                search_center=truth,
                search_radius=10_000,
            ),
            bearings,
            ranges,
        )

    baseline, _, _ = solve_with(1)
    scaled, scaled_bearings, scaled_ranges = solve_with(factor)
    _require(baseline.status is FixStatus.CONVERGED, "baseline uncertainty failed")
    _require(scaled.status is FixStatus.CONVERGED, "scaled uncertainty failed")
    _require(distance(baseline.position, scaled.position) < 0.01, "scale moved exact fix")
    ratio = scaled.uncertainty.semi_major_95 / baseline.uncertainty.semi_major_95
    _require(abs(ratio - factor) < 1e-6, "uncertainty ellipse did not scale")

    return {
        "id": "uncertainty-ellipse",
        "title": "Larger uncertainty, wider ellipse",
        "lesson": (
            "Multiplying every observation uncertainty by 8.5 leaves this exact "
            "fix in place and multiplies both confidence axes by 8.5."
        ),
        "schematic_note": (
            "Ellipses are local, linearized 95% confidence estimates in east/north "
            "metres. Both are drawn at 15× scale for legibility; the diagnostic "
            "values are unscaled. They are not safety bounds."
        ),
        "references": _reference_points(references, truth),
        "observations": _observations(scaled_bearings, scaled_ranges),
        "positions": [
            _point("truth", "Actual boat", "truth", truth, truth),
            _point("fix", "NautiPy fix", "fix", scaled.position, truth),
        ],
        "ellipses": [
            _uncertainty_payload(
                baseline,
                identifier="baseline-uncertainty",
                label="Original uncertainty",
                display_scale=15.0,
            ),
            _uncertainty_payload(
                scaled,
                identifier="scaled-uncertainty",
                label="Uncertainty × 8.5",
                display_scale=15.0,
            ),
        ],
        "diagnostics": {
            **_diagnostics(scaled),
            "uncertainty_scale": factor,
            "baseline_semi_major_95_m": _rounded(
                baseline.uncertainty.semi_major_95,
                3,
            ),
            "scaled_semi_major_95_m": _rounded(
                scaled.uncertainty.semi_major_95,
                3,
            ),
        },
    }


def build_document() -> dict[str, Any]:
    """Return the complete deterministic browser fixture."""

    scenarios = [
        _two_range_scenario(),
        _third_range_scenario(),
        _tangent_range_scenario(),
        _bearing_scenario(weak=False),
        _bearing_scenario(weak=True),
        _noisy_mixed_scenario(),
        _weighting_scenario(),
        _uncertainty_scale_scenario(),
    ]
    _require(
        len({scenario["id"] for scenario in scenarios}) == len(scenarios),
        "scenario identifiers are not unique",
    )
    return {
        "schema_version": 1,
        "earth_model": "WGS84",
        "generated_by": "website/tools/generate_fix_lab.py using NautiPy public API",
        "browser_model": (
            "Precomputed results only; the browser draws local schematics and "
            "does not solve a position."
        ),
        "safety_note": (
            "Educational demonstration only. NautiPy is not certified navigation "
            "equipment."
        ),
        "scenarios": scenarios,
    }


def render_document() -> str:
    """Return stable, human-readable JSON with a final newline."""

    return json.dumps(build_document(), indent=2, ensure_ascii=False) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"fixture path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the committed fixture is stale",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the generated JSON without writing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rendered = render_document()
    if args.stdout:
        sys.stdout.write(rendered)
        return 0

    output = args.output.resolve()
    current = output.read_text(encoding="utf-8") if output.exists() else None
    if current == rendered:
        print(f"Verified {output}")
        return 0
    if args.check:
        print(
            f"{output} is missing or stale; run "
            "python website/tools/generate_fix_lab.py",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
