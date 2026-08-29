import numpy as np

import pycolmap


def image_from_world(camera_center: np.ndarray) -> np.ndarray:
    intrinsic = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    camera_from_world = np.eye(4, dtype=np.float32)
    camera_from_world[:3, 3] = -camera_center
    return intrinsic @ camera_from_world[:3]


def project(projection: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.append(point, np.float32(1.0))
    projected = projection @ homogeneous
    return projected[:2] / projected[2]


def test_row_point_refinement_uses_coordinate_join_and_complete_chunks():
    expected = np.array(((0.2, -0.1, 4.0), (-0.3, 0.25, 5.0)), np.float32)
    projections = np.stack(
        tuple(
            image_from_world(center)
            for center in np.array(
                ((-0.8, 0.0, 0.0), (0.0, 0.4, 0.0), (0.9, -0.1, 0.0)),
                np.float32,
            )
        )
    )
    first_join = pycolmap.RowTrackSource(
        np.array((0, 2, 4), np.int64),
        np.array((0, 1, 0, 2), np.int32),
        np.array(
            ((10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)),
            np.float32,
        ),
        np.array((0, 1, 2), np.int32),
    )
    second_join = pycolmap.RowTrackSource(
        np.array((0, 2, 4), np.int64),
        np.array((2, 1, 2, 1), np.int32),
        np.array(
            ((40.05, 40.0), (60.0, 60.0), (50.0, 50.0), (20.0, 20.05)),
            np.float32,
        ),
        np.array((0, 1, 2), np.int32),
    )
    joined = pycolmap.join_row_tracks((first_join, second_join))

    np.testing.assert_array_equal(joined.duplicate_observation_indices, (0, 3))
    second_duplicate_indices = joined.duplicate_observation_indices[
        joined.duplicate_observation_offsets[
            1
        ] : joined.duplicate_observation_offsets[2]
    ]

    first_xy = np.stack(
        (
            project(projections[0], expected[0]),
            project(projections[1], expected[0]),
            project(projections[0], expected[1]),
            project(projections[2], expected[1]),
        )
    ).astype(np.float32)
    second_xy = np.stack(
        (
            project(projections[2], expected[1]),
            project(projections[1], expected[1]),
            project(projections[2], expected[0]),
            project(projections[1], expected[0]),
        )
    ).astype(np.float32)
    identity_transforms = np.tile(np.eye(4), (3, 1, 1))
    first = pycolmap.CasparRowPointRefinementSource(
        expected + np.array((0.2, -0.15, 0.3), np.float32),
        np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), np.float32),
        np.array((0, 1), np.int64),
        np.array((0, 2, 4), np.int64),
        np.array((0, 1, 0, 2), np.int32),
        first_xy,
        np.empty(0, np.uint32),
        np.array((0, 1, 2), np.uint32),
        identity_transforms,
    )
    source_translation = np.eye(4)
    source_translation[:3, 3] = np.array((0.4, -0.2, 0.1))
    second = pycolmap.CasparRowPointRefinementSource(
        (expected[::-1] - source_translation[:3, 3]).astype(np.float32),
        np.array(((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)), np.float32),
        np.array((0, 1), np.int64),
        np.array((0, 2, 4), np.int64),
        np.array((2, 1, 2, 1), np.int32),
        second_xy,
        second_duplicate_indices,
        np.array((0, 1, 2), np.uint32),
        np.tile(source_translation, (3, 1, 1)),
    )
    options = pycolmap.CasparPointRefinementOptions()
    options.solver_iter_max = 20
    result = pycolmap.caspar_refine_row_points(
        (first, second),
        joined.point_track_offsets,
        joined.point_track_indices,
        joined.point_observation_counts,
        projections,
        np.array((0,), np.uint32),
        (expected[0] + np.array((0.05, -0.04, 0.1), np.float32))[None],
        1,
        2,
        options,
    )

    np.testing.assert_array_equal(joined.point_observation_counts, (3, 3))
    np.testing.assert_allclose(result.points, expected, atol=2e-4)
    np.testing.assert_allclose(
        result.colors,
        ((1.0, 0.5, 0.0), (0.0, 0.5, 0.5)),
    )
    assert result.chunk_count == 2
