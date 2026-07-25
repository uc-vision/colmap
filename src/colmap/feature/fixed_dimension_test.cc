// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/feature/fixed_dimension.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#if defined(COLMAP_CUDA_ENABLED)
#include "colmap/util/cuda.h"
#endif

#include <gtest/gtest.h>

namespace colmap {
namespace {

FeatureDescriptors CreateDescriptors(const int descriptor_dimension) {
  FeatureDescriptorsFloat descriptors;
  descriptors.type = FeatureExtractorType::FIXED_DIMENSION;
  descriptors.data = FeatureDescriptorsFloatData::Zero(2, descriptor_dimension);
  const int first_dimension = descriptor_dimension - 8;
  descriptors.data.block(0, first_dimension, 1, 4).setConstant(0.5f);
  descriptors.data.block(1, first_dimension + 4, 1, 4).setConstant(0.5f);
  return descriptors.ToBytes();
}

FeatureDescriptors ReverseDescriptors(const FeatureDescriptors& descriptors) {
  FeatureDescriptorsFloat reversed = descriptors.ToFloat();
  reversed.data = reversed.data.colwise().reverse().eval();
  return reversed.ToBytes();
}

void ExpectReversedMatches(const FeatureMatches& matches) {
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], FeatureMatch(0, 1));
  EXPECT_EQ(matches[1], FeatureMatch(1, 0));
}

std::unique_ptr<FeatureMatcher> CreateMatcher(const bool use_gpu) {
  FeatureMatchingOptions options(
      FeatureMatcherType::FIXED_DIMENSION_BRUTEFORCE);
  options.use_gpu = use_gpu;
  options.gpu_index = "0";
  options.max_num_matches = 100;
  return THROW_CHECK_NOTNULL(CreateFixedDimensionFeatureMatcher(options));
}

void TestDescriptorDimension(const int descriptor_dimension,
                             const bool use_gpu) {
  const FeatureDescriptors descriptors1 =
      CreateDescriptors(descriptor_dimension);
  const FeatureDescriptors descriptors2 = ReverseDescriptors(descriptors1);
  const FeatureMatcher::Image image1 = {
      1,
      nullptr,
      nullptr,
      std::make_shared<FeatureDescriptors>(descriptors1),
  };
  const FeatureMatcher::Image image2 = {
      2,
      nullptr,
      nullptr,
      std::make_shared<FeatureDescriptors>(descriptors2),
  };

  FeatureMatches matches;
  CreateMatcher(use_gpu)->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);
}

void TestGuidedMatchingWithDistortion(const bool use_gpu) {
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[3] = 0.5;
  camera.params[4] = -0.5;
  camera.params[5] = 0.5;
  camera.params[6] = -0.5;

  const Eigen::Vector2f image_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f image_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f image_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f image_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureDescriptors descriptors1 = CreateDescriptors(256);
  const FeatureMatcher::Image image1 = {
      1,
      &camera,
      std::make_shared<FeatureKeypoints>(
          FeatureKeypoints{{image_point11.x(), image_point11.y()},
                           {image_point12.x(), image_point12.y()}}),
      std::make_shared<FeatureDescriptors>(descriptors1),
  };
  const FeatureMatcher::Image image2 = {
      2,
      &camera,
      std::make_shared<FeatureKeypoints>(
          FeatureKeypoints{{image_point21.x(), image_point21.y()},
                           {image_point22.x(), image_point22.y()}}),
      std::make_shared<FeatureDescriptors>(ReverseDescriptors(descriptors1)),
  };

  TwoViewGeometry two_view_geometry;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     *two_view_geometry.E,
                                     camera.CalibrationMatrix());

  auto matcher = CreateMatcher(use_gpu);
  constexpr double kMaxError = 1.0;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  EXPECT_TRUE(two_view_geometry.inlier_matches.empty());

  two_view_geometry.config = TwoViewGeometry::CALIBRATED_RIG;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedMatches(two_view_geometry.inlier_matches);
}

TEST(FixedDimensionFeatureMatcherCPU, SupportsArbitraryDescriptorDimensions) {
  TestDescriptorDimension(73, false);
  TestDescriptorDimension(128, false);
  TestDescriptorDimension(256, false);
}

TEST(FixedDimensionFeatureMatcherCPU, GuidedMatchingUsesCalibratedCoordinates) {
  TestGuidedMatchingWithDistortion(false);
}

#if defined(COLMAP_CUDA_ENABLED)
TEST(FixedDimensionFeatureMatcherGPU,
     MatchesCPUAtArbitraryDescriptorDimensions) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP();
  }
  TestDescriptorDimension(73, true);
  TestDescriptorDimension(128, true);
  TestDescriptorDimension(256, true);
}

TEST(FixedDimensionFeatureMatcherGPU, GuidedMatchingUsesCalibratedCoordinates) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP();
  }
  TestGuidedMatchingWithDistortion(true);
}
#endif

}  // namespace
}  // namespace colmap
