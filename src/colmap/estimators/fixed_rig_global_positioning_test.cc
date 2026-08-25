#include "colmap/estimators/fixed_rig_global_positioning.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FixedRigGlobalPositioning, RecoversMetricRigPositions) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction ground_truth;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 2;
  dataset_options.num_cameras_per_rig = 3;
  dataset_options.num_frames_per_rig = 5;
  dataset_options.num_points3D = 200;
  dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(dataset_options, &ground_truth, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);
  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  Reconstruction reconstruction = ground_truth;
  for (const auto& frame_entry : reconstruction.Frames()) {
    const frame_t frame_id = frame_entry.first;
    Frame& mutable_frame = reconstruction.Frame(frame_id);
    mutable_frame.SetRigFromWorld(Rigid3d(
        mutable_frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  ASSERT_TRUE(RunFixedRigGlobalPositioning(options,
                                           FixedRigGlobalPositionerOptions(),
                                           pose_graph,
                                           reconstruction,
                                           database_cache.PosePriors(),
                                           /*min_tri_angle_deg=*/1.));

  EXPECT_THAT(ground_truth,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.5,
                                 /*max_scale_error=*/0.05,
                                 /*num_obs_tolerance=*/0.0));
}

}  // namespace
}  // namespace colmap
