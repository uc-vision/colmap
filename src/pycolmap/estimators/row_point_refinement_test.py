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
        observation_offsets=np.array((0, 2, 4), np.int64),
        observation_image_indices=np.array((0, 1, 0, 2), np.int32),
        observation_xy=np.array(
            ((10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)),
            np.float32,
        ),
        image_to_shared_index=np.array((0, 1, 2), np.int32),
    )
    second_join = pycolmap.RowTrackSource(
        observation_offsets=np.array((0, 2, 4), np.int64),
        observation_image_indices=np.array((2, 1, 2, 1), np.int32),
        observation_xy=np.array(
            ((40.75, 40.0), (60.0, 60.0), (50.0, 50.0), (20.0, 20.75)),
            np.float32,
        ),
        image_to_shared_index=np.array((0, 1, 2), np.int32),
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
    first_points = np.asfortranarray(
        expected + np.array((0.2, -0.15, 0.3), np.float32)
    )
    first_tracks = pycolmap.CasparRowTrackSource(
        first_points,
        np.array((0, 1), np.int64),
        np.array((0, 2, 4), np.int64),
        np.array((0, 1, 0, 2), np.int32),
        first_xy,
        np.empty(0, np.uint32),
        np.array((0, 1, 2), np.uint32),
    )
    assert np.shares_memory(first_tracks.points, first_points)
    assert first_tracks.points.strides == first_points.strides
    first_colors = np.asfortranarray(
        np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), np.float32)
    )
    first = pycolmap.CasparRowPointRefinementSource(
        first_tracks,
        first_colors,
        identity_transforms,
    )
    assert np.shares_memory(first.colors, first_colors)
    assert first.colors.strides == first_colors.strides
    source_translation = np.eye(4)
    source_translation[:3, 3] = np.array((0.4, -0.2, 0.1))
    second_tracks = pycolmap.CasparRowTrackSource(
        np.asfortranarray(
            (expected[::-1] - source_translation[:3, 3]).astype(np.float32)
        ),
        np.array((0, 1), np.int64),
        np.array((0, 2, 4), np.int64),
        np.array((2, 1, 2, 1), np.int32),
        second_xy,
        second_duplicate_indices,
        np.array((0, 1, 2), np.uint32),
    )
    second = pycolmap.CasparRowPointRefinementSource(
        second_tracks,
        np.asfortranarray(
            np.array(((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)), np.float32)
        ),
        np.tile(source_translation, (3, 1, 1)),
    )
    override = np.array((7.0, 8.0, 9.0), dtype=np.float32)
    selected_row_point_indices = np.arange(2, dtype=np.uint32)
    initialization = pycolmap.caspar_initialize_row_points(
        (first, second),
        joined.point_track_offsets,
        joined.point_track_indices,
        selected_row_point_indices,
        np.array((1,), dtype=np.uint32),
        override[None],
    )
    assert np.shares_memory(
        initialization.row_point_indices, selected_row_point_indices
    )
    np.testing.assert_array_equal(initialization.row_point_indices, (0, 1))
    np.testing.assert_allclose(
        initialization.points[0],
        expected[0] + np.array((0.1, -0.075, 0.15), dtype=np.float32),
    )
    np.testing.assert_array_equal(initialization.points[1], override)

    options = pycolmap.CasparRowPointRefinementOptions()
    options.solver_iter_max = 20
    options.validate_reprojection = True
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
    point_errors = np.array(
        tuple(
            np.mean(
                tuple(
                    np.linalg.norm(
                        project(projections[image], result.points[point])
                        - project(projections[image], expected[point])
                    )
                    for image in range(3)
                )
            )
            for point in range(2)
        )
    )
    assert result.reprojection.point_count == 2
    assert result.reprojection.observation_count == 6
    np.testing.assert_allclose(
        result.reprojection.mean_pixels, np.mean(point_errors)
    )
    np.testing.assert_allclose(
        result.reprojection.median_pixels, np.median(point_errors)
    )
    np.testing.assert_allclose(
        result.reprojection.p95_pixels, np.percentile(point_errors, 95)
    )
    assert result.validation_seconds > 0.0
