import numpy as np
import pytest

import pycolmap


def test_view_graph_calibration_options_init():
    options = pycolmap.ViewGraphCalibrationOptions()
    assert options is not None


def test_global_mapper_options_init():
    options = pycolmap.GlobalMapperOptions()
    options.retriangulation_max_num_refinements = 2
    options.retriangulation_ba_max_num_iterations = 25

    assert options.retriangulation_max_num_refinements == 2
    assert options.retriangulation_ba_max_num_iterations == 25


def test_global_pipeline_options_init():
    options = pycolmap.GlobalPipelineOptions()
    assert options is not None


def test_global_pipeline_options_min_num_matches():
    options = pycolmap.GlobalPipelineOptions()
    options.min_num_matches = 20
    assert options.min_num_matches == 20


def test_prepared_global_mapping_finishes_with_refined_keypoints(database):
    pycolmap.set_random_seed(0)
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 4
    dataset_options.num_points3D = 50
    dataset_options.two_view_geometry_has_relative_pose = False
    pycolmap.synthesize_dataset(dataset_options, database)

    options = pycolmap.GlobalPipelineOptions()
    options.min_num_matches = 2
    options.num_threads = 1
    options.random_seed = 0
    options.mapper.global_positioning.use_gpu = False
    options.mapper.bundle_adjustment.ceres.use_gpu = False

    prepared = pycolmap.prepare_global_mapping(database, options)
    tracks = prepared.track_arrays()
    selected = np.array([0, len(tracks["observation_xy"]) - 1])
    image_ids = tracks["observation_image_ids"][selected]
    point2D_indices = tracks["observation_point2D_indices"][selected]
    refined = tracks["observation_xy"][selected] + np.array(
        [[0.25, -0.25], [-0.125, 0.375]], dtype=np.float32
    )

    reconstruction = prepared.finish(image_ids, point2D_indices, refined)

    assert reconstruction.num_points3D() > 0
    for image_id, point2D_idx, xy in zip(
        image_ids, point2D_indices, refined, strict=True
    ):
        np.testing.assert_allclose(
            reconstruction.image(int(image_id)).point2D(int(point2D_idx)).xy,
            xy,
        )


@pytest.mark.parametrize(
    "name",
    [
        "incremental_mapping",
        "global_mapping",
        "triangulate_points",
        "calibrate_view_graph",
        "bundle_adjustment",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))
