import math
import unittest

from nautipy import (
    AmbiguousCoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
    Position,
)
from nautipy.geojson import (
    from_geojson_feature_collection,
    from_geojson_point,
    to_geojson_feature_collection,
    to_geojson_point,
)


class PositionMetadataTests(unittest.TestCase):
    def test_metadata_is_keyword_only_and_not_part_of_equality(self) -> None:
        plain = Position(50, 8)
        described = Position(
            50,
            8,
            identifier="station-1",
            description="Harbour station",
        )

        self.assertEqual(plain, described)
        self.assertEqual(hash(plain), hash(described))
        self.assertEqual(described.identifier, "station-1")
        self.assertEqual(described.description, "Harbour station")
        with self.assertRaises(TypeError):
            Position(50, 8, "station-1")  # type: ignore[misc]

    def test_metadata_types_are_validated(self) -> None:
        for identifier in (True, [], object()):
            with self.subTest(identifier=identifier):
                with self.assertRaises(CoordinateParseError):
                    Position(50, 8, identifier=identifier)  # type: ignore[arg-type]
        for identifier in (math.inf, -math.inf, math.nan):
            with self.subTest(identifier=identifier):
                with self.assertRaises(CoordinateRangeError):
                    Position(50, 8, identifier=identifier)
        with self.assertRaises(CoordinateParseError):
            Position(50, 8, description=3)  # type: ignore[arg-type]


class GeoJSONPointTests(unittest.TestCase):
    def test_point_round_trip_uses_longitude_latitude_order(self) -> None:
        mapping = to_geojson_point("50.12257, 8.66570")

        self.assertEqual(
            mapping,
            {"type": "Point", "coordinates": [8.66570, 50.12257]},
        )
        self.assertEqual(from_geojson_point(mapping), Position(50.12257, 8.66570))

    def test_point_export_delegates_position_parsing_options(self) -> None:
        self.assertEqual(
            to_geojson_point([8, 50], order="lonlat"),
            {"type": "Point", "coordinates": [8.0, 50.0]},
        )
        self.assertEqual(
            to_geojson_point(
                {"latitude": "50° 0′ N", "longitude": "8° 0′ E"},
                format="ddm",
            ),
            {"type": "Point", "coordinates": [8.0, 50.0]},
        )

    def test_point_output_normalizes_negative_zero(self) -> None:
        self.assertEqual(
            to_geojson_point(Position(-0.0, -0.0))["coordinates"],
            [0.0, 0.0],
        )

    def test_bare_point_rejects_metadata_that_it_cannot_preserve(self) -> None:
        with self.assertRaisesRegex(
            CoordinateParseError,
            "feature_collection",
        ):
            to_geojson_point(Position(50, 8, identifier="station-1"))

    def test_point_foreign_cross_type_and_crs_members(self) -> None:
        self.assertEqual(
            from_geojson_point(
                {
                    "type": "Point",
                    "coordinates": [8, 50],
                    "bbox": [8, 50, 8, 50],
                    "vendor": "value",
                }
            ),
            Position(50, 8),
        )
        for member, value in (
            ("features", []),
            ("geometries", []),
            ("geometry", None),
            ("properties", {}),
        ):
            with self.subTest(member=member):
                with self.assertRaisesRegex(CoordinateParseError, member):
                    from_geojson_point(
                        {"type": "Point", "coordinates": [8, 50], member: value}
                    )
        with self.assertRaisesRegex(CoordinateParseError, "CRS"):
            from_geojson_point(
                {"type": "Point", "coordinates": [8, 50], "crs": {}}
            )

    def test_point_rejects_unsupported_and_malformed_geometry(self) -> None:
        with self.assertRaisesRegex(CoordinateParseError, "LineString"):
            from_geojson_point(
                {
                    "type": "LineString",
                    "coordinates": [[8, 50], [9, 51]],
                }
            )
        with self.assertRaisesRegex(CoordinateParseError, "exactly two"):
            from_geojson_point(
                {"type": "Point", "coordinates": [8, 50, 100]}
            )
        with self.assertRaises(CoordinateRangeError):
            from_geojson_point(
                {"type": "Point", "coordinates": [181, 50]}
            )


class GeoJSONFeatureCollectionTests(unittest.TestCase):
    def test_round_trip_preserves_order_identifier_and_description(self) -> None:
        positions = (
            Position(
                50,
                8,
                identifier="station-1",
                description="Harbour station",
            ),
            Position(-1, -2, identifier=0, description=""),
            Position(0, 0),
        )

        mapping = to_geojson_feature_collection(position for position in positions)
        self.assertEqual(mapping["type"], "FeatureCollection")
        features = mapping["features"]
        self.assertIsInstance(features, list)
        self.assertEqual(
            features,
            [
                {
                    "type": "Feature",
                    "id": "station-1",
                    "geometry": {"type": "Point", "coordinates": [8.0, 50.0]},
                    "properties": {"description": "Harbour station"},
                },
                {
                    "type": "Feature",
                    "id": 0,
                    "geometry": {"type": "Point", "coordinates": [-2.0, -1.0]},
                    "properties": {"description": ""},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    "properties": {},
                },
            ],
        )

        restored = from_geojson_feature_collection(mapping)
        self.assertEqual(restored, positions)
        self.assertEqual(
            [(item.identifier, item.description) for item in restored],
            [("station-1", "Harbour station"), (0, ""), (None, None)],
        )

    def test_empty_collection_and_null_properties_are_supported(self) -> None:
        self.assertEqual(
            from_geojson_feature_collection(
                {"type": "FeatureCollection", "features": []}
            ),
            (),
        )
        result = from_geojson_feature_collection(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [8, 50]},
                        "properties": None,
                    }
                ],
            }
        )
        self.assertEqual(result, (Position(50, 8),))

    def test_empty_collection_still_validates_parsing_options(self) -> None:
        with self.assertRaisesRegex(CoordinateParseError, "order"):
            to_geojson_feature_collection([], order="bogus")  # type: ignore[arg-type]
        with self.assertRaisesRegex(CoordinateParseError, "format"):
            to_geojson_feature_collection([], format="bogus")

    def test_foreign_members_are_accepted_but_not_preserved(self) -> None:
        result = from_geojson_feature_collection(
            {
                "type": "FeatureCollection",
                "bbox": [8, 50, 8, 50],
                "vendor": "collection",
                "features": [
                    {
                        "type": "Feature",
                        "vendor": "feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [8, 50],
                            "vendor": "geometry",
                        },
                        "properties": {
                            "description": "Known",
                            "unknown": {"nested": True},
                        },
                    }
                ],
            }
        )

        self.assertEqual(result[0].description, "Known")
        self.assertEqual(
            to_geojson_feature_collection(result)["features"][0]["properties"],
            {"description": "Known"},
        )

    def test_every_unsupported_geometry_is_rejected_without_skipping(self) -> None:
        geometry_coordinates = {
            "LineString": [[8, 50], [9, 51]],
            "MultiPoint": [[8, 50]],
            "Polygon": [[[8, 50], [9, 50], [8, 50]]],
            "MultiLineString": [[[8, 50], [9, 51]]],
            "MultiPolygon": [[[[8, 50], [9, 50], [8, 50]]]],
        }
        for geometry_type, coordinates in geometry_coordinates.items():
            with self.subTest(geometry_type=geometry_type):
                collection = self._collection(
                    {"type": geometry_type, "coordinates": coordinates}
                )
                with self.assertRaisesRegex(
                    CoordinateParseError,
                    rf"features\[0\]\.geometry.*{geometry_type}",
                ):
                    from_geojson_feature_collection(collection)
        with self.assertRaisesRegex(CoordinateParseError, "GeometryCollection"):
            from_geojson_feature_collection(
                self._collection(
                    {"type": "GeometryCollection", "geometries": []}
                )
            )
        with self.assertRaisesRegex(CoordinateParseError, "null"):
            from_geojson_feature_collection(self._collection(None))

    def test_cross_type_members_and_crs_are_rejected_at_every_level(self) -> None:
        collection_members = ("coordinates", "geometries", "geometry", "properties")
        for member in collection_members:
            with self.subTest(level="collection", member=member):
                value = {"type": "FeatureCollection", "features": [], member: []}
                with self.assertRaisesRegex(CoordinateParseError, member):
                    from_geojson_feature_collection(value)

        for member in ("coordinates", "features", "geometries"):
            with self.subTest(level="feature", member=member):
                feature = self._feature({"type": "Point", "coordinates": [8, 50]})
                feature[member] = []
                with self.assertRaisesRegex(CoordinateParseError, member):
                    from_geojson_feature_collection(
                        {"type": "FeatureCollection", "features": [feature]}
                    )

        for level in ("collection", "feature", "geometry"):
            with self.subTest(level=level):
                value = self._collection({"type": "Point", "coordinates": [8, 50]})
                if level == "collection":
                    value["crs"] = {}
                elif level == "feature":
                    value["features"][0]["crs"] = {}
                else:
                    value["features"][0]["geometry"]["crs"] = {}
                with self.assertRaisesRegex(CoordinateParseError, "CRS"):
                    from_geojson_feature_collection(value)

    def test_collection_rejects_malformed_shape_and_metadata(self) -> None:
        invalid_collections = (
            {},
            {"type": "Point", "features": []},
            {"type": "FeatureCollection"},
            {"type": "FeatureCollection", "features": "not an array"},
            {"type": "FeatureCollection", "features": [{}]},
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": None, "properties": {}}
                ],
            },
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [8, 50]},
                    }
                ],
            },
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [8, 50]},
                        "properties": [],
                    }
                ],
            },
        )
        for value in invalid_collections:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    from_geojson_feature_collection(value)

        for identifier in (None, True, [], {}, math.nan, math.inf):
            with self.subTest(identifier=identifier):
                feature = self._feature({"type": "Point", "coordinates": [8, 50]})
                feature["id"] = identifier
                with self.assertRaises((CoordinateParseError, CoordinateRangeError)):
                    from_geojson_feature_collection(
                        {"type": "FeatureCollection", "features": [feature]}
                    )

        for description in (1, True, [], {}):
            with self.subTest(description=description):
                feature = self._feature({"type": "Point", "coordinates": [8, 50]})
                feature["properties"] = {"description": description}
                with self.assertRaises(CoordinateParseError):
                    from_geojson_feature_collection(
                        {"type": "FeatureCollection", "features": [feature]}
                    )

    def test_collection_export_rejects_a_single_position_input(self) -> None:
        for value in (Position(50, 8), "50, 8", {"lat": 50, "lon": 8}):
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    to_geojson_feature_collection(value)  # type: ignore[arg-type]

    def test_collection_export_preserves_ambiguity_diagnostics(self) -> None:
        with self.assertRaises(AmbiguousCoordinateError) as context:
            to_geojson_feature_collection([[8, 50]], order="auto")

        self.assertIn("positions[0]", str(context.exception))
        self.assertTrue(context.exception.candidates)

    @staticmethod
    def _feature(geometry: object) -> dict[str, object]:
        return {"type": "Feature", "geometry": geometry, "properties": {}}

    @classmethod
    def _collection(cls, geometry: object) -> dict[str, object]:
        return {
            "type": "FeatureCollection",
            "features": [cls._feature(geometry)],
        }


if __name__ == "__main__":
    unittest.main()
