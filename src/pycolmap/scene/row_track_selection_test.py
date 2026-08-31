import numpy as np
import pycolmap


def test_row_track_selection_reports_quota_plateau() -> None:
    source = pycolmap.RowTrackSelectionSource(
        np.asarray((0, 1, 2, 3, 4), dtype=np.int64),
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=np.int32),
        np.asarray(((10, 10), (10, 10), (80, 80), (10, 10)), dtype=np.float32),
        np.asarray((0, 0, 1, 2), dtype=np.uint32),
        np.zeros(1, dtype=np.int32),
        np.asarray(((100, 100),), dtype=np.float32),
    )
    support = np.ones(3, dtype=np.uint16)

    selected = [
        pycolmap.select_row_bundle_points((source,), support, 1, density)
        for density in (1, 2, 3)
    ]

    np.testing.assert_array_equal(selected[0].point_indices, (0, 1))
    np.testing.assert_array_equal(selected[1].point_indices, (0, 1))
    np.testing.assert_array_equal(selected[2].point_indices, (0, 1, 2))
    assert [result.interior_quota_truncated for result in selected] == [
        True,
        True,
        False,
    ]
