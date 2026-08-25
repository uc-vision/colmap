import numpy as np

import pycolmap


def test_packed_rig_calibration_matches_legacy():
    synthetic = pycolmap.SyntheticDatasetOptions(
        num_rigs=1,
        num_cameras_per_rig=3,
        num_frames_per_rig=6,
        num_points3D=12,
        num_points2D_without_point3D=0,
    )
    reconstruction = pycolmap.synthesize_dataset(synthetic)
    frame_ids = sorted(reconstruction.reg_frame_ids())

    legacy_groups = []
    rigs_from_group = []
    frame0_to_frame2_distances = []
    group_track_offsets = [0]
    track_observation_offsets = [0]
    xyz = []
    frame_indices = []
    camera_ids = []
    xy = []
    distance_stddev = 0.01

    for group_start in range(0, len(frame_ids), 3):
        group_frame_ids = frame_ids[group_start : group_start + 3]
        frame_index_by_id = {
            frame_id: index for index, frame_id in enumerate(group_frame_ids)
        }
        poses = [
            reconstruction.frame(frame_id).rig_from_world
            for frame_id in group_frame_ids
        ]
        distance = float(
            np.linalg.norm(
                poses[2].tgt_origin_in_src() - poses[0].tgt_origin_in_src()
            )
        )
        tracks = []
        for point in reconstruction.points3D.values():
            observations = []
            track_frame_indices = []
            track_camera_ids = []
            track_xy = []
            for element in point.track.elements:
                image = reconstruction.image(element.image_id)
                frame_index = frame_index_by_id.get(image.frame_id)
                if frame_index is None:
                    continue
                point_xy = image.point2D(element.point2D_idx).xy
                observations.append(
                    pycolmap.RigCalibrationObservation(
                        frame_idx=frame_index,
                        camera_id=image.camera_id,
                        xy=point_xy,
                    )
                )
                track_frame_indices.append(frame_index)
                track_camera_ids.append(image.camera_id)
                track_xy.append(point_xy)
            if len(observations) < 2:
                continue
            tracks.append(
                pycolmap.RigCalibrationTrack(
                    xyz=point.xyz,
                    observations=observations,
                )
            )
            xyz.append(point.xyz)
            frame_indices.extend(track_frame_indices)
            camera_ids.extend(track_camera_ids)
            xy.extend(track_xy)
            track_observation_offsets.append(len(frame_indices))

        legacy_groups.append(
            pycolmap.RigCalibrationGroup(
                rigs_from_group=poses,
                tracks=tracks,
                frame0_to_frame2_distance=pycolmap.RigCalibrationDistancePrior(
                    distance=distance,
                    stddev=distance_stddev,
                ),
            )
        )
        rigs_from_group.append([pose.matrix() for pose in poses])
        frame0_to_frame2_distances.append(distance)
        group_track_offsets.append(len(xyz))

    options = pycolmap.RigCalibrationOptions(
        refine_focal_length=False,
        refine_principal_point=False,
        refine_distortion=False,
        refine_sensor_from_rig=False,
        print_summary=False,
    )
    rig_id = next(iter(reconstruction.rigs.keys()))
    legacy_summary = pycolmap.create_ceres_rig_calibrator(
        options,
        rig_id,
        legacy_groups,
        pycolmap.Reconstruction(reconstruction),
    ).solve()
    packed_summary = pycolmap.create_ceres_rig_calibrator_packed(
        options,
        rig_id,
        pycolmap.Reconstruction(reconstruction),
        np.asarray(rigs_from_group, dtype=np.float64),
        np.asarray(frame0_to_frame2_distances, dtype=np.float64),
        distance_stddev,
        np.asarray(group_track_offsets, dtype=np.uint64),
        np.asarray(track_observation_offsets, dtype=np.uint64),
        np.asarray(xyz, dtype=np.float64),
        np.asarray(frame_indices, dtype=np.uint8),
        np.asarray(camera_ids, dtype=np.uint32),
        np.asarray(xy, dtype=np.float64),
    ).solve()

    count_fields = (
        "num_groups",
        "num_tracks",
        "num_observations",
        "num_filtered_groups",
        "num_filtered_observations",
        "num_invalid_observations",
    )
    assert tuple(
        getattr(packed_summary, field) for field in count_fields
    ) == tuple(getattr(legacy_summary, field) for field in count_fields)
    np.testing.assert_allclose(
        packed_summary.reprojection_errors,
        legacy_summary.reprojection_errors,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        packed_summary.distance_prior_errors,
        legacy_summary.distance_prior_errors,
        rtol=0.0,
        atol=1e-12,
    )
