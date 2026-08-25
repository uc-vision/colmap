// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/feature_matching.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FixedRigMatching, GuidesAllIntraFramePairsFromSensorExtrinsics) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 3;
  dataset_options.num_frames_per_rig = 1;
  dataset_options.num_points3D = 50;
  dataset_options.num_points2D_without_point3D = 0;
  dataset_options.camera_has_prior_focal_length = true;
  dataset_options.prior_position = true;
  SynthesizeDataset(dataset_options, &reconstruction, database.get());
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SpatialPairingOptions pairing_options;
  pairing_options.max_num_neighbors = 1;
  pairing_options.min_num_neighbors = 1;
  pairing_options.max_distance = 1e6;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;
  matching_options.use_fixed_rig_geometry = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto matcher = CreateSpatialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 3);
  ASSERT_EQ(database->ReadAllMatches().size(), 3);
  ASSERT_EQ(database->ReadTwoViewGeometries().size(), 3);

  const Rig rig = database->ReadAllRigs().front();
  const auto camera_from_rig = [&rig](const Image& image) {
    return rig.IsRefSensor(image.DataId().sensor_id)
               ? Rigid3d()
               : rig.SensorFromRig(image.DataId().sensor_id);
  };
  for (size_t i = 0; i < images.size(); ++i) {
    for (size_t j = i + 1; j < images.size(); ++j) {
      const FeatureMatches matches =
          database->ReadMatches(images[i].ImageId(), images[j].ImageId());
      const TwoViewGeometry geometry = database->ReadTwoViewGeometry(
          images[i].ImageId(), images[j].ImageId());
      const Rigid3d expected_cam2_from_cam1 =
          camera_from_rig(images[j]) * Inverse(camera_from_rig(images[i]));

      EXPECT_GE(matches.size(), geometry_options.min_num_inliers);
      EXPECT_EQ(matches, geometry.inlier_matches);
      EXPECT_EQ(geometry.config,
                TwoViewGeometry::ConfigurationType::CALIBRATED_RIG);
      EXPECT_EQ(geometry.cam2_from_cam1, expected_cam2_from_cam1);
      ASSERT_TRUE(geometry.E.has_value());
      EXPECT_TRUE(geometry.E->isApprox(
          EssentialMatrixFromPose(expected_cam2_from_cam1)));
    }
  }
}

}  // namespace
}  // namespace colmap
