import numpy as np

import pycolmap


def test_packed_rig_calibration_handoff():
    reconstruction = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(
            num_rigs=1,
            num_cameras_per_rig=3,
            num_frames_per_rig=3,
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
    distance = float(
        np.linalg.norm(
            poses[2].tgt_origin_in_src() - poses[0].tgt_origin_in_src()
        )
    )
    summary = pycolmap.create_ceres_rig_calibrator(
        options,
        next(iter(reconstruction.rigs)),
        pycolmap.Reconstruction(reconstruction),
        np.asarray([[pose.matrix() for pose in poses]], dtype=np.float64),
        np.asarray([distance], dtype=np.float64),
        0.01,
        np.asarray([0, len(xyz)], dtype=np.uint64),
        np.asarray(track_observation_offsets, dtype=np.uint64),
        np.asarray(xyz, dtype=np.float64),
        np.asarray(frame_indices, dtype=np.uint8),
        np.asarray(camera_ids, dtype=np.uint32),
        np.asarray(xy, dtype=np.float64),
    ).solve()

    assert summary.is_solution_usable()
    assert summary.num_groups == 1
    assert summary.num_tracks == len(xyz)
    assert summary.num_observations == len(frame_indices)
