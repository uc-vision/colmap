import numpy as np

import pycolmap


def test_packed_rig_calibration_handoff_with_four_frames():
    reconstruction = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(
            num_rigs=1,
            num_cameras_per_rig=3,
            num_frames_per_rig=4,
            num_points3D=12,
            num_points2D_without_point3D=0,
        )
    )
    frame_ids = sorted(reconstruction.reg_frame_ids())
    frame_index_by_id = {
        frame_id: index for index, frame_id in enumerate(frame_ids)
    }
    poses = [
        reconstruction.frame(frame_id).rig_from_world for frame_id in frame_ids
    ]

    xyz = []
    frame_indices = []
    camera_ids = []
    xy = []
    track_observation_offsets = [0]
    for point in reconstruction.points3D.values():
        observations = [
            (
                frame_index_by_id[
                    reconstruction.image(element.image_id).frame_id
                ],
                element,
            )
            for element in point.track.elements
        ]
        if len(observations) < 2:
            continue
        xyz.append(point.xyz)
        for frame_index, element in observations:
            image = reconstruction.image(element.image_id)
            frame_indices.append(frame_index)
            camera_ids.append(image.camera_id)
            xy.append(image.point2D(element.point2D_idx).xy)
        track_observation_offsets.append(len(frame_indices))

    options = pycolmap.RigCalibrationOptions(
        refine_focal_length=False,
        refine_principal_point=False,
        refine_distortion=False,
        refine_sensor_from_rig=False,
        print_summary=False,
    )
    options.ceres.use_gpu = False
    first_to_last_distance = float(
        np.linalg.norm(
            poses[-1].tgt_origin_in_src() - poses[0].tgt_origin_in_src()
        )
    )
    summary = pycolmap.create_ceres_rig_calibrator(
        options=options,
        rig_id=next(iter(reconstruction.rigs)),
        reconstruction=pycolmap.Reconstruction(reconstruction),
        rigs_from_group=np.asarray(
            [[pose.matrix() for pose in poses]], dtype=np.float64
        ),
        first_to_last_distances=np.asarray(
            [first_to_last_distance], dtype=np.float64
        ),
        distance_stddev=0.01,
        group_track_offsets=np.asarray([0, len(xyz)], dtype=np.uint64),
        track_observation_offsets=np.asarray(
            track_observation_offsets, dtype=np.uint64
        ),
        xyz=np.asarray(xyz, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.uint32),
        camera_ids=np.asarray(camera_ids, dtype=np.uint32),
        xy=np.asarray(xy, dtype=np.float64),
    ).solve()

    assert summary.is_solution_usable()
    assert summary.num_groups == 1
    assert summary.num_tracks == len(xyz)
    assert summary.num_observations == len(frame_indices)


def test_four_frame_packed_preparation_retries_robust_track_to_cap():
    pycolmap.set_random_seed(0)
    reconstruction = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(
            num_rigs=1,
            num_cameras_per_rig=3,
            num_frames_per_rig=4,
            num_points3D=4,
            num_points2D_without_point3D=0,
        )
    )
    rig = next(iter(reconstruction.rigs.values()))
    frame_ids = sorted(reconstruction.reg_frame_ids())
    frame_index_by_id = {
        frame_id: index for index, frame_id in enumerate(frame_ids)
    }
    rigs_from_group = np.asarray(
        [
            reconstruction.frame(frame_id).rig_from_world.matrix()
            for frame_id in frame_ids
        ],
        dtype=np.float64,
    )
    image_ids = np.asarray(sorted(reconstruction.images), dtype=np.uint32)
    image_frame_indices = np.asarray(
        [
            frame_index_by_id[reconstruction.image(int(image_id)).frame_id]
            for image_id in image_ids
        ],
        dtype=np.uint32,
    )
    image_camera_ids = np.asarray(
        [
            reconstruction.image(int(image_id)).camera_id
            for image_id in image_ids
        ],
        dtype=np.uint32,
    )
    points = [point for _, point in sorted(reconstruction.points3D.items())]
    point_observations = []
    for point in points:
        xy_by_image = {
            element.image_id: reconstruction.image(element.image_id)
            .point2D(element.point2D_idx)
            .xy
            for element in point.track.elements
        }
        point_observations.append(
            np.asarray([xy_by_image[int(image_id)] for image_id in image_ids])
        )
    keypoints_by_image = np.stack(point_observations, axis=1)

    first_camera = reconstruction.camera(int(image_camera_ids[0]))
    width = first_camera.width
    height = first_camera.height
    # The first candidate has no coherent multi-view geometry and must fail.
    keypoints_by_image[:, 0] = np.asarray(
        [
            ((index + 1) * width * 10, -(index + 1) * height * 10)
            for index in range(len(image_ids))
        ]
    )
    # Two gross outliers put the next linear seed behind the cameras, while
    # its seven consistent observations let robust triangulation recover it.
    keypoints_by_image[0, 1] = (-12 * width, -20 * height)
    keypoints_by_image[1, 1] = (15 * width, 0.5 * height)

    cams_from_group = []
    cameras = []
    for frame_index, camera_id in zip(
        image_frame_indices, image_camera_ids, strict=True
    ):
        sensor_id = pycolmap.sensor_t(
            pycolmap.SensorType.CAMERA, int(camera_id)
        )
        sensor_from_rig = (
            pycolmap.Rigid3d()
            if rig.is_ref_sensor(sensor_id)
            else rig.sensor_from_rig(sensor_id)
        )
        cams_from_group.append(
            sensor_from_rig
            * reconstruction.frame(frame_ids[int(frame_index)]).rig_from_world
        )
        cameras.append(reconstruction.camera(int(camera_id)))
    camera_rays = np.asarray(
        [
            camera.cam_ray_from_img(xy)
            for camera, xy in zip(
                cameras, keypoints_by_image[:, 1], strict=True
            )
        ]
    )
    linear_xyz = pycolmap.triangulate_multi_view_point(
        [cam_from_group.matrix() for cam_from_group in cams_from_group],
        camera_rays,
    )
    assert linear_xyz is not None
    assert any(
        (cam_from_group * linear_xyz)[2] <= np.finfo(np.float64).eps
        for cam_from_group in cams_from_group
    )

    num_images = len(image_ids)
    num_components = len(points)
    result = pycolmap.prepare_rig_calibration_group(
        reconstruction,
        rig.rig_id,
        np.arange(
            0,
            (num_components + 1) * num_images,
            num_images,
            dtype=np.uint32,
        ),
        np.asarray(
            [
                (int(image_id) << 32) | component_index
                for component_index in range(num_components)
                for image_id in image_ids
            ],
            dtype=np.uint64,
        ),
        np.arange(num_components, dtype=np.uint32),
        2,
        image_ids,
        image_frame_indices,
        image_camera_ids,
        np.arange(
            0,
            (num_images + 1) * num_components,
            num_components,
            dtype=np.uint32,
        ),
        np.ascontiguousarray(keypoints_by_image.reshape(-1, 2)),
        rigs_from_group,
    )

    assert result.attempted_tracks == 3
    assert result.retained_tracks == 2
    np.testing.assert_array_equal(
        result.track_observation_offsets, [0, num_images, 2 * num_images]
    )
    np.testing.assert_allclose(result.xyz, [points[1].xyz, points[2].xyz])
    np.testing.assert_allclose(
        result.xy,
        np.concatenate((keypoints_by_image[:, 1], keypoints_by_image[:, 2])),
    )
    assert result.frame_indices[0] == result.frame_indices[1]
    assert result.frame_indices.dtype == np.uint32
    assert result.camera_ids[0] != result.camera_ids[1]
