#include "colmap/sfm/global_mapper.h"

#include "colmap/math/math.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Add tests for pose priors.

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

class NoOpPositioningStrategy final : public GlobalMapperStrategy {
 public:
  bool RunPositioning(const GlobalPositionerOptions&,
                      const PoseGraph&,
                      Reconstruction&,
                      const std::vector<PosePrior>&,
                      double) const override {
    return true;
  }
};

TEST(GlobalMapper, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(GlobalMapper, RelaxesAngularFilteringWithoutFocalPrior) {
  SetPRNGSeed(0);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 5;
  synthetic_dataset_options.camera_model_id = PinholeCameraModel::model_id;
  synthetic_dataset_options.camera_params = {1280, 1280, 512, 384};
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper global_mapper(CreateDatabaseCache(*database),
                             std::make_shared<NoOpPositioningStrategy>());
  global_mapper.BeginReconstruction(reconstruction);
  *reconstruction = gt_reconstruction;

  std::vector<camera_t> camera_ids;
  for (const auto& [camera_id, _] : reconstruction->Cameras()) {
    camera_ids.push_back(camera_id);
  }
  ASSERT_EQ(camera_ids.size(), 2);
  std::sort(camera_ids.begin(), camera_ids.end());
  const camera_t prior_camera_id = camera_ids[0];
  const camera_t estimated_camera_id = camera_ids[1];
  reconstruction->Camera(prior_camera_id).has_prior_focal_length = true;
  reconstruction->Camera(estimated_camera_id).has_prior_focal_length = false;

  const auto& [point3D_id, point3D] = *reconstruction->Points3D().begin();
  TrackElement prior_observation;
  TrackElement estimated_observation;
  bool found_prior_observation = false;
  bool found_estimated_observation = false;
  for (const TrackElement& observation : point3D.track.Elements()) {
    const camera_t camera_id =
        reconstruction->Image(observation.image_id).CameraId();
    if (camera_id == prior_camera_id && !found_prior_observation) {
      prior_observation = observation;
      found_prior_observation = true;
    } else if (camera_id == estimated_camera_id &&
               !found_estimated_observation) {
      estimated_observation = observation;
      found_estimated_observation = true;
    }
  }
  ASSERT_TRUE(found_prior_observation);
  ASSERT_TRUE(found_estimated_observation);
  ASSERT_GE(point3D.track.Length(), 4);

  reconstruction->Image(prior_observation.image_id)
      .Point2D(prior_observation.point2D_idx)
      .xy.x() += 1.0;
  reconstruction->Image(estimated_observation.image_id)
      .Point2D(estimated_observation.point2D_idx)
      .xy.x() += 1.0;

  const auto angular_error_deg = [&](const TrackElement& observation) {
    const Image& image = reconstruction->Image(observation.image_id);
    return RadToDeg(CalculateAngularReprojectionError(
        image.Point2D(observation.point2D_idx).xy,
        reconstruction->Point3D(point3D_id).xyz,
        image.CamFromWorld(),
        *image.CameraPtr()));
  };
  const double prior_error = angular_error_deg(prior_observation);
  const double estimated_error = angular_error_deg(estimated_observation);
  const double min_error = std::min(prior_error, estimated_error);
  const double max_error = std::max(prior_error, estimated_error);
  ASSERT_LT(max_error, 2.0 * min_error);
  const double angular_threshold = (0.5 * max_error + min_error) / 2.0;
  ASSERT_GT(prior_error, angular_threshold);
  ASSERT_GT(estimated_error, angular_threshold);
  ASSERT_LT(estimated_error, 2.0 * angular_threshold);

  ASSERT_TRUE(
      global_mapper.GlobalPositioning(GlobalPositionerOptions(),
                                      angular_threshold,
                                      /*max_normalized_reproj_error=*/1.0,
                                      /*min_tri_angle_deg=*/0.0));

  EXPECT_FALSE(reconstruction->Image(prior_observation.image_id)
                   .Point2D(prior_observation.point2D_idx)
                   .HasPoint3D());
  EXPECT_TRUE(reconstruction->Image(estimated_observation.image_id)
                  .Point2D(estimated_observation.point2D_idx)
                  .HasPoint3D());
}

TEST(GlobalMapperOptions, RefineSensorFromRigPropagatesToSubOptions) {
  GlobalMapperOptions options;
  options.refine_sensor_from_rig = false;
  // Sub-options keep their own defaults (true) until accessed.
  EXPECT_TRUE(options.rotation_averaging.refine_sensor_from_rig);
  EXPECT_TRUE(options.global_positioning.refine_sensor_from_rig);
  EXPECT_TRUE(options.bundle_adjustment.refine_sensor_from_rig);
  // Accessors return resolved sub-options with the top-level flag applied.
  EXPECT_FALSE(options.RotationAveraging().refine_sensor_from_rig);
  EXPECT_FALSE(options.GlobalPositioning().refine_sensor_from_rig);
  EXPECT_FALSE(options.BundleAdjustment().refine_sensor_from_rig);
}

}  // namespace
}  // namespace colmap
