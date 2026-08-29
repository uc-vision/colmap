import numpy as np

import pycolmap


def test_bundle_adjustment_termination_type_enum():
    assert {
        k: int(v)
        for k, v in pycolmap.BundleAdjustmentTerminationType.__members__.items()
    } == {
        "CONVERGENCE": 0,
        "NO_CONVERGENCE": 1,
        "FAILURE": 2,
        "USER_SUCCESS": 3,
        "USER_FAILURE": 4,
    }


def test_bundle_adjustment_gauge_enum():
    assert {
        k: int(v) for k, v in pycolmap.BundleAdjustmentGauge.__members__.items()
    } == {
        "UNSPECIFIED": -1,
        "TWO_CAMS_FROM_WORLD": 0,
        "THREE_POINTS": 1,
    }


def test_bundle_adjustment_backend_enum():
    assert {
        k: int(v)
        for k, v in pycolmap.BundleAdjustmentBackend.__members__.items()
    } == {
        "CERES": 0,
        "CASPAR": 1,
    }
    assert (
        pycolmap.BundleAdjustmentBackend("CASPAR")
        == pycolmap.BundleAdjustmentBackend.CASPAR
    )


def test_loss_function_type_enum():
    assert {
        k: int(v) for k, v in pycolmap.LossFunctionType.__members__.items()
    } == {
        "TRIVIAL": 0,
        "SOFT_L1": 1,
        "CAUCHY": 2,
        "HUBER": 3,
    }


def test_bundle_adjustment_summary_default_init():
    summary = pycolmap.BundleAdjustmentSummary()
    assert summary is not None


def test_bundle_adjustment_summary_num_residuals_readwrite():
    summary = pycolmap.BundleAdjustmentSummary()
    summary.num_residuals = 100
    assert summary.num_residuals == 100


def test_bundle_adjustment_summary_termination_type_readwrite():
    summary = pycolmap.BundleAdjustmentSummary()
    summary.termination_type = (
        pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
    )
    assert (
        summary.termination_type
        == pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
    )


def test_bundle_adjustment_summary_is_solution_usable():
    summary = pycolmap.BundleAdjustmentSummary()
    result = summary.is_solution_usable()
    assert isinstance(result, bool)


def test_bundle_adjustment_summary_brief_report():
    summary = pycolmap.BundleAdjustmentSummary()
    report = summary.brief_report()
    assert isinstance(report, str)


def test_ceres_bundle_adjustment_summary_default_init():
    summary = pycolmap.CeresBundleAdjustmentSummary()
    assert summary is not None


def test_ceres_bundle_adjustment_summary_inherits():
    summary = pycolmap.CeresBundleAdjustmentSummary()
    assert isinstance(summary, pycolmap.BundleAdjustmentSummary)


def test_bundle_adjustment_config_default_init():
    config = pycolmap.BundleAdjustmentConfig()
    assert config is not None


def test_bundle_adjustment_config_images_property():
    config = pycolmap.BundleAdjustmentConfig()
    images = config.images
    assert len(images) == 0


def test_ceres_ba_options_default_init():
    options = pycolmap.CeresBundleAdjustmentOptions()
    assert options is not None


def test_ceres_ba_options_check():
    options = pycolmap.CeresBundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_caspar_ba_options_default_init():
    options = pycolmap.CasparBundleAdjustmentOptions()
    assert options is not None


def test_caspar_ba_options_readwrite():
    options = pycolmap.CasparBundleAdjustmentOptions()
    options.solver_iter_max = 17
    options.pcg_iter_max = 11
    options.diag_init = 2.0
    options.gpu_index = "0"
    assert options.solver_iter_max == 17
    assert options.pcg_iter_max == 11
    assert options.diag_init == 2.0
    assert options.gpu_index == "0"


def test_ba_options_default_init():
    options = pycolmap.BundleAdjustmentOptions()
    assert options is not None


def test_ba_options_refine_focal_length_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_focal_length
    assert isinstance(original, bool)
    options.refine_focal_length = not original
    assert options.refine_focal_length == (not original)


def test_ba_options_refine_principal_point_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_principal_point
    assert isinstance(original, bool)
    options.refine_principal_point = not original
    assert options.refine_principal_point == (not original)


def test_ba_options_refine_extra_params_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_extra_params
    assert isinstance(original, bool)
    options.refine_extra_params = not original
    assert options.refine_extra_params == (not original)


def test_ba_options_refine_points3d_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_points3D
    assert isinstance(original, bool)
    options.refine_points3D = not original
    assert options.refine_points3D == (not original)


def test_ba_options_min_track_length_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    assert isinstance(options.min_track_length, int)
    options.min_track_length = 5
    assert options.min_track_length == 5


def test_ba_options_print_summary_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.print_summary
    assert isinstance(original, bool)
    options.print_summary = not original
    assert options.print_summary == (not original)


def test_ba_options_backend_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    options.backend = pycolmap.BundleAdjustmentBackend.CERES
    assert options.backend == pycolmap.BundleAdjustmentBackend.CERES
    options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
    assert options.backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_ba_options_ceres_property():
    options = pycolmap.BundleAdjustmentOptions()
    ceres = options.ceres
    assert isinstance(ceres, pycolmap.CeresBundleAdjustmentOptions)


def test_ba_options_caspar_property():
    options = pycolmap.BundleAdjustmentOptions()
    caspar = options.caspar
    assert isinstance(caspar, pycolmap.CasparBundleAdjustmentOptions)


def test_ba_options_check():
    options = pycolmap.BundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_pose_prior_ba_options_default_init():
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    assert options is not None


def test_pose_prior_ba_options_prior_position_fallback_stddev_readwrite():
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    assert isinstance(options.prior_position_fallback_stddev, float)
    options.prior_position_fallback_stddev = 5.0
    assert options.prior_position_fallback_stddev == 5.0


def test_ceres_pose_prior_ba_options_default_init():
    options = pycolmap.CeresPosePriorBundleAdjustmentOptions()
    assert options is not None


def test_ceres_pose_prior_ba_options_check():
    options = pycolmap.CeresPosePriorBundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_create_default_bundle_adjuster(synthetic_reconstruction):
    options = pycolmap.BundleAdjustmentOptions()
    config = pycolmap.BundleAdjustmentConfig()
    adjuster = pycolmap.create_default_bundle_adjuster(
        options, config, synthetic_reconstruction
    )
    assert adjuster is not None


def test_create_default_ceres_bundle_adjuster(synthetic_reconstruction):
    options = pycolmap.BundleAdjustmentOptions()
    config = pycolmap.BundleAdjustmentConfig()
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        options, config, synthetic_reconstruction
    )
    assert adjuster is not None


def test_bundle_adjustment_pipeline(synthetic_reconstruction):
    reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    options = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    pycolmap.bundle_adjustment(reconstruction, options)


def pinhole_projections(camera_centers):
    projections = np.zeros((len(camera_centers), 3, 4), dtype=np.float32)
    projections[:, :3, :3] = np.array(
        [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    projections[:, 0, 3] = -400.0 * np.asarray(camera_centers, dtype=np.float32)
    return projections


def project_observations(projections, points, point_indices, image_indices):
    homogeneous_points = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    projected = np.einsum(
        "nij,nj->ni",
        projections[image_indices],
        homogeneous_points[point_indices],
    )
    return np.ascontiguousarray(projected[:, :2] / projected[:, 2:3])


def test_caspar_refines_points_against_exact_fixed_cameras():
    projections = pinhole_projections([-1.0, 0.0, 1.0])
    original_projections = projections.copy()
    expected = np.array([[0.2, -0.1, 4.0], [-0.4, 0.3, 5.0]], dtype=np.float32)
    initial = np.array([[0.7, 0.2, 3.2], [-0.8, 0.0, 4.1]], dtype=np.float32)
    point_indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint32)
    image_indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32)
    pixels = project_observations(
        projections, expected, point_indices, image_indices
    )

    options = pycolmap.CasparPointRefinementOptions()
    options.loss_scale = 100.0
    options.solver_iter_max = 100
    result = pycolmap.caspar_refine_pinhole_points(
        initial,
        projections,
        point_indices,
        image_indices,
        pixels,
        options,
    )

    np.testing.assert_allclose(result.points, expected, atol=1e-3)
    np.testing.assert_array_equal(projections, original_projections)
    assert result.summary.final_score < result.summary.initial_score


def test_caspar_point_refinement_soft_l1_rejects_outlier():
    projections = pinhole_projections([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([[0.2, -0.1, 4.0]], dtype=np.float32)
    initial = np.array([[0.5, 0.1, 3.5]], dtype=np.float32)
    point_indices = np.zeros(5, dtype=np.uint32)
    image_indices = np.arange(5, dtype=np.uint32)
    pixels = project_observations(
        projections, expected, point_indices, image_indices
    )
    pixels[-1] += np.array([300.0, -200.0], dtype=np.float32)

    options = pycolmap.CasparPointRefinementOptions()
    options.loss_scale = 1.0
    options.solver_iter_max = 100
    robust = pycolmap.caspar_refine_pinhole_points(
        initial,
        projections,
        point_indices,
        image_indices,
        pixels,
        options,
    ).points[0]
    options.loss_scale = 1e6
    quadratic = pycolmap.caspar_refine_pinhole_points(
        initial,
        projections,
        point_indices,
        image_indices,
        pixels,
        options,
    ).points[0]

    robust_error = np.linalg.norm(robust - expected[0])
    quadratic_error = np.linalg.norm(quadratic - expected[0])
    assert robust_error < 0.03
    assert robust_error < 0.2 * quadratic_error
