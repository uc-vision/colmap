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

#include "colmap/estimators/fixed_rig_two_view_geometry.h"

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <future>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestData {
  Rig rig1;
  Rig rig2;
  std::vector<FixedRigMatchedPair> pairs;
  Reconstruction reconstruction;
};

TestData CreateTestData() {
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 3;
  options.num_frames_per_rig = 1;
  options.num_points3D = 200;
  options.inlier_match_ratio = 1.0;
  options.camera_has_prior_focal_length = true;

  TestData data;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(options, &data.reconstruction, database.get());
  data.rig1 = data.reconstruction.Rig(1);
  data.rig2 = data.reconstruction.Rig(2);

  for (auto& [pair_id, matches] : database->ReadAllMatches()) {
    auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Image* image1 = &data.reconstruction.Image(image_id1);
    const Image* image2 = &data.reconstruction.Image(image_id2);
    const Camera* camera1 = &data.reconstruction.Camera(image1->CameraId());
    const Camera* camera2 = &data.reconstruction.Camera(image2->CameraId());
    if (data.rig1.HasSensor(camera2->SensorId()) &&
        data.rig2.HasSensor(camera1->SensorId())) {
      std::swap(image_id1, image_id2);
      std::swap(image1, image2);
      std::swap(camera1, camera2);
      for (FeatureMatch& match : matches) {
        std::swap(match.point2D_idx1, match.point2D_idx2);
      }
    }
    if (!data.rig1.HasSensor(camera1->SensorId()) ||
        !data.rig2.HasSensor(camera2->SensorId())) {
      continue;
    }

    FixedRigMatchedPair pair;
    pair.image_id1 = image_id1;
    pair.image_id2 = image_id2;
    pair.camera1 = *camera1;
    pair.camera2 = *camera2;
    pair.matches = std::move(matches);
    pair.points1.reserve(pair.matches.size());
    pair.points2.reserve(pair.matches.size());
    for (const FeatureMatch& match : pair.matches) {
      pair.points1.push_back(image1->Point2D(match.point2D_idx1).xy);
      pair.points2.push_back(image2->Point2D(match.point2D_idx2).xy);
    }
    data.pairs.push_back(std::move(pair));
  }
  return data;
}

TEST(EstimateFixedRigTwoViewGeometries, BoundedRansacScoresAllMatches) {
  SetPRNGSeed(1);
  const TestData data = CreateTestData();

  TwoViewGeometryOptions options;
  options.ransac_options.random_seed = 42;
  constexpr size_t kMaxNumRansacMatches = 64;
  auto concurrent_estimation = std::async(std::launch::async, [&]() {
    return EstimateFixedRigTwoViewGeometries(
        data.rig1, data.rig2, data.pairs, options, kMaxNumRansacMatches);
  });
  const auto geometries = EstimateFixedRigTwoViewGeometries(
      data.rig1, data.rig2, data.pairs, options, kMaxNumRansacMatches);
  const auto concurrent_geometries = concurrent_estimation.get();

  ASSERT_EQ(geometries.size(), data.pairs.size());
  ASSERT_EQ(concurrent_geometries.size(), geometries.size());
  size_t num_inliers = 0;
  for (size_t i = 0; i < geometries.size(); ++i) {
    const auto& [image_pair, geometry] = geometries[i];
    EXPECT_EQ(concurrent_geometries[i].first, image_pair);
    EXPECT_EQ(concurrent_geometries[i].second.inlier_matches,
              geometry.inlier_matches);
    EXPECT_EQ(geometry.config,
              TwoViewGeometry::ConfigurationType::CALIBRATED_RIG);
    ASSERT_TRUE(geometry.cam2_from_cam1.has_value());
    EXPECT_THAT(
        *geometry.cam2_from_cam1,
        Rigid3dNear(
            data.reconstruction.Image(image_pair.second).CamFromWorld() *
                Inverse(
                    data.reconstruction.Image(image_pair.first).CamFromWorld()),
            /*rtol=*/1e-2,
            /*ttol=*/1e-2));
    num_inliers += geometry.inlier_matches.size();
  }
  EXPECT_GT(num_inliers, kMaxNumRansacMatches);
}

}  // namespace
}  // namespace colmap
