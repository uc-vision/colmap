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

#include "colmap/feature/types.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateTestDatabase(int num_images, Database& database) {
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_images;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.num_points2D_without_point3D = 3;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, &database);
}

std::unique_ptr<retrieval::VisualIndex> CreateSyntheticVisualIndex() {
  auto visual_index = retrieval::VisualIndex::Create();
  retrieval::VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  visual_index->Build(
      build_options,
      FeatureDescriptorsFloat(FeatureExtractorType::SIFT,
                              FeatureDescriptorsFloatData::Random(50, 128)));
  return visual_index;
}

TEST(CreateExhaustiveFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  ExhaustivePairingOptions pairing_options;
  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateExhaustiveFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_EQ(database->ReadAllMatches().size(), 6);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 6);
}

TEST(CreateVocabTreeFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto vocab_tree_path = test_dir / "vocab_tree.bin";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  // Create vocab tree
  CreateSyntheticVisualIndex()->Write(vocab_tree_path);

  VocabTreePairingOptions pairing_options;
  pairing_options.vocab_tree_path = vocab_tree_path;
  pairing_options.num_images = 2;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateVocabTreeFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Each image should match with num_images others,
  // while some of the pairs may be redundant.
  EXPECT_GE(database->ReadAllMatches().size(), 4);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 4);
}

TEST(CreateSequentialFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/5, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 2;
  pairing_options.quadratic_overlap = false;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSequentialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // With 5 images and overlap=2:
  // (0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)
  EXPECT_EQ(database->ReadAllMatches().size(), 7);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 7);
}

TEST(CreateSpatialFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SpatialPairingOptions pairing_options;
  pairing_options.max_num_neighbors = 2;
  pairing_options.max_distance = 1e6;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSpatialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_GT(database->ReadAllMatches().size(), 0);
  EXPECT_GT(database->ReadTwoViewGeometries().size(), 0);
}

TEST(CreateSpatialFeatureMatcher, FixedRigGuidedMatching) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SpatialPairingOptions pairing_options;
  pairing_options.max_num_neighbors = 1;
  pairing_options.min_num_neighbors = 1;
  pairing_options.max_distance = 1e6;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;
  matching_options.guided_matching = true;
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
  auto camera_from_rig = [&rig](const Image& image) {
    return rig.IsRefSensor(image.DataId().sensor_id)
               ? Rigid3d()
               : rig.SensorFromRig(image.DataId().sensor_id);
  };

  for (size_t i = 0; i < images.size(); ++i) {
    for (size_t j = i + 1; j < images.size(); ++j) {
      const FeatureMatches matches =
          database->ReadMatches(images[i].ImageId(), images[j].ImageId());
      const TwoViewGeometry two_view_geometry = database->ReadTwoViewGeometry(
          images[i].ImageId(), images[j].ImageId());
      const Rigid3d expected_cam2_from_cam1 =
          camera_from_rig(images[j]) * Inverse(camera_from_rig(images[i]));

      EXPECT_GE(matches.size(), geometry_options.min_num_inliers);
      EXPECT_EQ(matches, two_view_geometry.inlier_matches);
      EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::CALIBRATED_RIG);
      EXPECT_EQ(two_view_geometry.cam2_from_cam1, expected_cam2_from_cam1);
      ASSERT_TRUE(two_view_geometry.E.has_value());
      EXPECT_TRUE(two_view_geometry.E->isApprox(
          EssentialMatrixFromPose(expected_cam2_from_cam1)));
    }
  }
}

TEST(CreateTransitiveFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);

  // Create initial matches: 1-2 and 2-3
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.inlier_matches = FeatureMatches(10);

  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[1].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[1].ImageId(), images[2].ImageId(), two_view_geometry);

  TransitivePairingOptions pairing_options;
  pairing_options.batch_size = 100;
  pairing_options.num_iterations = 1;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateTransitiveFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Should create transitive match 1-3
  const size_t final_matches = database->ReadTwoViewGeometries().size();
  EXPECT_GE(final_matches, 2);  // At least the original 2 matches
}

TEST(CreateImagePairsFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto match_list_path = test_dir / "match_list.txt";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);

  // Create match list file with specific image pairs
  std::ofstream file(match_list_path);
  file << images[0].Name() << " " << images[1].Name() << "\n";
  file << images[1].Name() << " " << images[2].Name() << "\n";
  file << images[2].Name() << " " << images[3].Name() << "\n";
  file.close();

  ImportedPairingOptions pairing_options;
  pairing_options.match_list_path = match_list_path;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateImagePairsFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_EQ(database->ReadAllMatches().size(), 3);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 3);
}

TEST(CreateFeaturePairsFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto match_list_path = test_dir / "feature_match_list.txt";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/3, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 2);

  // Create feature match list file with many matches for better verification
  std::ofstream file(match_list_path);
  file << images[0].Name() << " " << images[1].Name() << "\n";
  for (int i = 0; i < 15; ++i) {
    file << i << " " << i << "\n";
  }
  file << "\n";  // Empty line separates pairs
  file << images[1].Name() << " " << images[2].Name() << "\n";
  for (int i = 0; i < 15; ++i) {
    file << i << " " << i << "\n";
  }
  file << "\n";
  file.close();

  FeaturePairsMatchingOptions pairing_options;
  pairing_options.match_list_path = match_list_path;
  pairing_options.verify_matches = true;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;  // Lower threshold for testing

  auto matcher = CreateFeaturePairsFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Should have imported and verified the matches
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 2);
}

TEST(CreateGeometricVerifier, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearTwoViewGeometries();

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  EXPECT_GE(database->ReadAllMatches().size(), 3);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 3);
}

void ExpectRigVerificationResults(const Database& database,
                                  int num_expected_matches,
                                  int num_expected_calibrated,
                                  int num_expected_calibrated_rig) {
  // Verify that two-view geometries were created.
  int num_calibrated = 0;
  int num_calibrated_rig = 0;
  int num_others = 0;
  for (const auto& [pair_id, two_view_geometry] :
       database.ReadTwoViewGeometries()) {
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), num_expected_matches);
    switch (two_view_geometry.config) {
      case TwoViewGeometry::CALIBRATED:
        ++num_calibrated;
        break;
      case TwoViewGeometry::CALIBRATED_RIG:
        ++num_calibrated_rig;
        break;
      default:
        ++num_others;
    }
  }
  // Two calibrated pairs between images in the same frames.
  EXPECT_EQ(num_calibrated, num_expected_calibrated);
  // Four calibrated pairs between images in different frames.
  EXPECT_EQ(num_calibrated_rig, num_expected_calibrated_rig);
  EXPECT_EQ(num_others, 0);
}

TEST(CreateGeometricVerifier, RigVerificationWithNonTrivialFrames) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 25;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = -1;
  verifier_options.rig_verification = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // All pairs should be overwritten with calibrated rig pairs.
  ExpectRigVerificationResults(*database,
                               synthetic_dataset_options.num_points3D,
                               /*num_expected_calibrated=*/0,
                               /*num_expected_calibrated_rig=*/15);
}

TEST(CreateGeometricVerifier, RigVerificationWithTrivialFrames) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 25;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  verifier_options.rig_verification = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // Trivial frames should be skipped and unmodified.
  ExpectRigVerificationResults(*database,
                               synthetic_dataset_options.num_points3D,
                               /*num_expected_calibrated=*/1,
                               /*num_expected_calibrated_rig=*/0);
}

TEST(RunFixedRigGeometricVerification, SoleCrossFrameVerifier) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  std::vector<frame_t> frame_ids;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    frame_ids.push_back(frame_id);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  ASSERT_EQ(frame_ids.size(), 3);

  std::optional<std::pair<image_t, image_t>> same_frame_pair;
  TwoViewGeometry same_frame_geometry;
  std::optional<std::pair<image_t, image_t>> singleton_pair;
  std::optional<std::pair<image_t, image_t>> orphan_geometry_pair;
  std::vector<std::pair<image_t, image_t>> failed_group_pairs;
  for (const auto& [pair_id, _] : database->ReadAllMatches()) {
    const auto image_pair = PairIdToImagePair(pair_id);
    frame_t frame_id1 = reconstruction.Image(image_pair.first).FrameId();
    frame_t frame_id2 = reconstruction.Image(image_pair.second).FrameId();
    if (frame_id1 > frame_id2) {
      std::swap(frame_id1, frame_id2);
    }
    if (frame_id1 == frame_id2) {
      if (!same_frame_pair.has_value()) {
        same_frame_pair = image_pair;
        same_frame_geometry =
            database->ReadTwoViewGeometry(image_pair.first, image_pair.second);
      }
      continue;
    }
    if (frame_id1 == frame_ids[0] && frame_id2 == frame_ids[1]) {
      continue;
    }
    if (frame_id1 == frame_ids[0] && frame_id2 == frame_ids[2] &&
        !singleton_pair.has_value()) {
      singleton_pair = image_pair;
      continue;
    }
    if (frame_id1 == frame_ids[0] && frame_id2 == frame_ids[2] &&
        !orphan_geometry_pair.has_value()) {
      database->DeleteMatches(image_pair.first, image_pair.second);
      orphan_geometry_pair = image_pair;
      continue;
    }
    if (frame_id1 == frame_ids[1] && frame_id2 == frame_ids[2] &&
        failed_group_pairs.size() < 2) {
      database->DeleteMatches(image_pair.first, image_pair.second);
      database->WriteMatches(image_pair.first,
                             image_pair.second,
                             FeatureMatches(3, FeatureMatch(0, 0)));
      failed_group_pairs.push_back(image_pair);
      continue;
    }
    database->DeleteMatches(image_pair.first, image_pair.second);
    database->DeleteTwoViewGeometry(image_pair.first, image_pair.second);
  }
  ASSERT_TRUE(same_frame_pair.has_value());
  ASSERT_TRUE(singleton_pair.has_value());
  ASSERT_TRUE(orphan_geometry_pair.has_value());
  ASSERT_EQ(failed_group_pairs.size(), 2);
  ASSERT_TRUE(database->ExistsTwoViewGeometry(singleton_pair->first,
                                              singleton_pair->second));
  ASSERT_TRUE(database->ExistsTwoViewGeometry(orphan_geometry_pair->first,
                                              orphan_geometry_pair->second));

  FixedRigGeometricVerificationOptions verifier_options;
  verifier_options.num_threads = 1;
  verifier_options.max_num_ransac_matches = 64;
  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;
  geometry_options.ransac_options.random_seed = 42;
  RunFixedRigGeometricVerification(
      database_path, verifier_options, geometry_options);

  EXPECT_FALSE(database->ExistsTwoViewGeometry(singleton_pair->first,
                                               singleton_pair->second));
  EXPECT_FALSE(database->ExistsTwoViewGeometry(orphan_geometry_pair->first,
                                               orphan_geometry_pair->second));
  for (const auto& image_pair : failed_group_pairs) {
    EXPECT_FALSE(
        database->ExistsTwoViewGeometry(image_pair.first, image_pair.second));
  }
  const TwoViewGeometry same_frame_geometry_after =
      database->ReadTwoViewGeometry(same_frame_pair->first,
                                    same_frame_pair->second);
  EXPECT_EQ(same_frame_geometry_after.config, same_frame_geometry.config);
  EXPECT_EQ(same_frame_geometry_after.inlier_matches,
            same_frame_geometry.inlier_matches);
  EXPECT_EQ(same_frame_geometry_after.cam2_from_cam1,
            same_frame_geometry.cam2_from_cam1);
  size_t num_same_frame = 0;
  size_t num_verified_cross_frame = 0;
  for (const auto& [pair_id, geometry] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    if (frame_id1 == frame_id2) {
      ++num_same_frame;
    } else {
      EXPECT_EQ(geometry.config, TwoViewGeometry::CALIBRATED_RIG);
      ++num_verified_cross_frame;
    }
  }
  EXPECT_EQ(num_same_frame, 9);
  EXPECT_EQ(num_verified_cross_frame, 9);
}

TEST(CreateGeometricVerifier, Guided) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Clear all inlier matches. cam2_from_cam1 is already gt from the synthesized
  // database.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> gt_two_view_geometries =
      database->ReadTwoViewGeometries();
  for (const auto& [pair_id, _] : gt_two_view_geometries) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    database->DeleteInlierMatches(image_id1, image_id2);
  }

  ExistingMatchedPairingOptions pairing_options;
  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  verifier_options.use_existing_relative_pose = true;

  TwoViewGeometryOptions geometry_options;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // Check validity after guided geometric verification.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> two_view_geometries =
      database->ReadTwoViewGeometries();
  EXPECT_GE(two_view_geometries.size(), gt_two_view_geometries.size());
  for (size_t i = 0; i < two_view_geometries.size(); ++i) {
    EXPECT_EQ(two_view_geometries[i].first, gt_two_view_geometries[i].first);
    EXPECT_EQ(two_view_geometries[i].second.cam2_from_cam1,
              gt_two_view_geometries[i].second.cam2_from_cam1);
    EXPECT_TRUE(gt_two_view_geometries[i].second.E.value().isApprox(
        two_view_geometries[i].second.E.value()));
    // Should at least have all the original inliers. Some generated outliers
    // can be accidentally inliers as well.
    EXPECT_GE(two_view_geometries[i].second.inlier_matches.size(),
              gt_two_view_geometries[i].second.inlier_matches.size());
  }
}

}  // namespace
}  // namespace colmap
