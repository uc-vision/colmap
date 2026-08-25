#include "colmap/estimators/fixed_rig_rotation_averaging.h"

#include "colmap/math/math.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FixedRigRotationAveraging, RejectsFramePairOutlier) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction ground_truth;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 3;
  dataset_options.num_frames_per_rig = 2;
  dataset_options.num_points3D = 50;
  dataset_options.sensor_from_rig_rotation_stddev = 20.;
  dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(dataset_options, &ground_truth, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);
  Reconstruction reconstruction;
  reconstruction.Load(database_cache);
  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  std::vector<image_pair_t> frame_pair_edges;
  for (const auto& pair : pose_graph.ValidEdges()) {
    const image_pair_t pair_id = pair.first;
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (reconstruction.Image(image_id1).FrameId() !=
        reconstruction.Image(image_id2).FrameId()) {
      frame_pair_edges.push_back(pair_id);
    }
  }
  ASSERT_GT(frame_pair_edges.size(), 2);

  const image_pair_t outlier_pair_id = frame_pair_edges.front();
  PoseGraph::Edge& outlier = pose_graph.Edges().at(outlier_pair_id);
  const Eigen::Vector3d translation_before =
      outlier.cam2_from_cam1.translation();
  outlier.cam2_from_cam1.rotation() =
      Eigen::AngleAxisd(DegToRad(30.), Eigen::Vector3d::UnitX()) *
      outlier.cam2_from_cam1.rotation();

  NodeHashMap<image_pair_t, Eigen::Quaterniond> rotations_before;
  for (const image_pair_t pair_id : frame_pair_edges) {
    rotations_before.emplace(
        pair_id, pose_graph.Edges().at(pair_id).cam2_from_cam1.rotation());
  }

  FilterFixedRigRotationOutliers(
      pose_graph, reconstruction, /*max_rotation_error_deg=*/10.);

  EXPECT_FALSE(pose_graph.IsValid(outlier_pair_id));
  EXPECT_EQ(pose_graph.Edges().at(outlier_pair_id).cam2_from_cam1.translation(),
            translation_before);
  for (const image_pair_t pair_id : frame_pair_edges) {
    EXPECT_EQ(pose_graph.Edges().at(pair_id).cam2_from_cam1.rotation(),
              rotations_before.at(pair_id));
  }
}

}  // namespace
}  // namespace colmap
