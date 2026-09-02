import numpy as np
import pycolmap


def test_build_feature_tracks():
    tracks = pycolmap.build_feature_tracks(
        np.array([1, 2, 1], dtype=np.uint32),
        np.array([2, 3, 3], dtype=np.uint32),
        np.array([0, 2, 3, 4], dtype=np.uint64),
        np.array([[0, 1], [2, 3], [1, 4], [0, 4]], dtype=np.uint32),
    )

    np.testing.assert_array_equal(
        tracks.observation_offsets,
        np.array([0, 3, 5], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        tracks.observation_codes,
        np.array(
            [
                (1 << 32) | 0,
                (2 << 32) | 1,
                (3 << 32) | 4,
                (1 << 32) | 2,
                (2 << 32) | 3,
            ],
            dtype=np.uint64,
        ),
    )


def test_build_feature_tracks_rejects_merges_repeating_an_image():
    tracks = pycolmap.build_feature_tracks(
        np.array([1, 1, 2], dtype=np.uint32),
        np.array([2, 3, 3], dtype=np.uint32),
        np.array([0, 1, 2, 3], dtype=np.uint64),
        np.array([[0, 0], [1, 0], [0, 0]], dtype=np.uint32),
    )

    np.testing.assert_array_equal(
        tracks.observation_offsets,
        np.array([0, 2, 4], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        tracks.observation_codes,
        np.array(
            [
                (1 << 32) | 0,
                (2 << 32) | 0,
                (1 << 32) | 1,
                (3 << 32) | 0,
            ],
            dtype=np.uint64,
        ),
    )
