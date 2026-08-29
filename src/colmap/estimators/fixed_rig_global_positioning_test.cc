#include "colmap/estimators/fixed_rig_global_positioning.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void TestRecoversMetricRigPositions(const int num_rigs,
                                    const int num_frames_per_rig,
                                    const bool require_frame_constraints) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction ground_truth;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = num_rigs;
  dataset_options.num_cameras_per_rig = 3;
  dataset_options.num_frames_per_rig = num_frames_per_rig;
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
  FixedRigGlobalPositionerOptions rig_options;
  rig_options.require_frame_constraints = require_frame_constraints;
  ASSERT_TRUE(RunFixedRigGlobalPositioning(options,
                                           rig_options,
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

TEST(FixedRigGlobalPositioning, RecoversMetricRigPositions) {
  TestRecoversMetricRigPositions(/*num_rigs=*/2,
                                 /*num_frames_per_rig=*/5,
                                 /*require_frame_constraints=*/false);
}

TEST(FixedRigGlobalPositioning,
     RecoversMetricRigPositionsWithRequiredFrameConstraints) {
  TestRecoversMetricRigPositions(/*num_rigs=*/1,
                                 /*num_frames_per_rig=*/10,
                                 /*require_frame_constraints=*/true);
}

TEST(FixedRigGlobalPositioning,
     RequiredFrameConstraintsAcceptCoveredDisconnectedComponents) {
  Reconstruction ground_truth;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 3;
  dataset_options.num_frames_per_rig = 10;
  dataset_options.num_points3D = 2000;
  SynthesizeDataset(dataset_options, &ground_truth);

  Reconstruction reconstruction = ground_truth;
  std::vector<frame_t> frame_ids = reconstruction.RegFrameIds();
  std::sort(frame_ids.begin(), frame_ids.end());
  const size_t split_index = frame_ids.size() / 2;
  FlatHashSet<frame_t> second_component(frame_ids.begin() + split_index,
                                        frame_ids.end());
  std::vector<std::pair<image_t, point2D_t>> observations_to_delete;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    const bool keep_second_component = point3D_id % 2 == 0;
    for (const TrackElement& observation : point3D.track.Elements()) {
      const frame_t frame_id =
          reconstruction.Image(observation.image_id).FrameId();
      if ((second_component.count(frame_id) > 0) != keep_second_component) {
        observations_to_delete.emplace_back(observation.image_id,
                                            observation.point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : observations_to_delete) {
    reconstruction.DeleteObservation(image_id, point2D_idx);
  }

  const FlatHashSet<frame_t> component_anchors{frame_ids.front(),
                                               frame_ids[split_index]};
  for (const frame_t frame_id : frame_ids) {
    if (component_anchors.count(frame_id) > 0) {
      continue;
    }
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  FixedRigGlobalPositionerOptions rig_options;
  rig_options.require_frame_constraints = true;
  ASSERT_TRUE(RunFixedRigGlobalPositioning(options,
                                           rig_options,
                                           PoseGraph(),
                                           reconstruction,
                                           {},
                                           /*min_tri_angle_deg=*/1.));
  for (const size_t component_begin : {size_t{0}, split_index}) {
    const Eigen::Vector3d ground_truth_anchor =
        ground_truth.Frame(frame_ids[component_begin])
            .RigFromWorld()
            .TgtOriginInSrc();
    const Eigen::Vector3d reconstruction_anchor =
        reconstruction.Frame(frame_ids[component_begin])
            .RigFromWorld()
            .TgtOriginInSrc();
    for (size_t index = component_begin; index < component_begin + split_index;
         ++index) {
      const Eigen::Vector3d ground_truth_offset =
          ground_truth.Frame(frame_ids[index]).RigFromWorld().TgtOriginInSrc() -
          ground_truth_anchor;
      const Eigen::Vector3d reconstruction_offset =
          reconstruction.Frame(frame_ids[index])
              .RigFromWorld()
              .TgtOriginInSrc() -
          reconstruction_anchor;
      EXPECT_LT((ground_truth_offset - reconstruction_offset).norm(), 0.5);
    }
  }
}

TEST(FixedRigGlobalPositioning, RequiredFrameConstraintsRejectUncoveredFrame) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 3;
  dataset_options.num_frames_per_rig = 5;
  dataset_options.num_points3D = 100;
  SynthesizeDataset(dataset_options, &reconstruction);

  const frame_t uncovered_frame_id = reconstruction.RegFrameIds().back();
  std::vector<std::pair<image_t, point2D_t>> observations_to_delete;
  for (const data_t& data_id :
       reconstruction.Frame(uncovered_frame_id).ImageIds()) {
    const Image& image = reconstruction.Image(data_id.id);
    if (image.IsRefInFrame()) {
      continue;
    }
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      if (image.Point2D(point2D_idx).HasPoint3D()) {
        observations_to_delete.emplace_back(image.ImageId(), point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : observations_to_delete) {
    reconstruction.DeleteObservation(image_id, point2D_idx);
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  FixedRigGlobalPositionerOptions rig_options;
  rig_options.require_frame_constraints = true;
  EXPECT_FALSE(RunFixedRigGlobalPositioning(options,
                                            rig_options,
                                            PoseGraph(),
                                            reconstruction,
                                            {},
                                            /*min_tri_angle_deg=*/0.));
}

}  // namespace
}  // namespace colmap
