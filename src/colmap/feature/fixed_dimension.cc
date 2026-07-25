// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/feature/fixed_dimension.h"

#if defined(COLMAP_CUDA_ENABLED)
#include "colmap/feature/fixed_dimension_cuda.h"
#include "colmap/util/cuda.h"
#endif
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace colmap {
namespace {

void ThrowCheckFixedDimensionFeatures(const FeatureMatcher::Image& image1,
                                      const FeatureMatcher::Image& image2,
                                      const bool check_keypoints = false) {
  THROW_CHECK_NOTNULL(image1.descriptors);
  THROW_CHECK_NOTNULL(image2.descriptors);
  THROW_CHECK_EQ(image1.descriptors->type,
                 FeatureExtractorType::FIXED_DIMENSION);
  THROW_CHECK_EQ(image2.descriptors->type,
                 FeatureExtractorType::FIXED_DIMENSION);
  THROW_CHECK_EQ(image1.descriptors->data.cols() % sizeof(Eigen::half), 0);
  THROW_CHECK_EQ(image2.descriptors->data.cols() % sizeof(Eigen::half), 0);
  const Eigen::Index descriptor_dimension =
      image1.descriptors->data.cols() / sizeof(Eigen::half);
  THROW_CHECK_EQ(descriptor_dimension,
                 image2.descriptors->data.cols() / sizeof(Eigen::half));
  THROW_CHECK_GT(descriptor_dimension, 0);
  if (check_keypoints) {
    THROW_CHECK_NOTNULL(image1.camera);
    THROW_CHECK_NOTNULL(image2.camera);
    THROW_CHECK_NOTNULL(image1.keypoints);
    THROW_CHECK_NOTNULL(image2.keypoints);
    THROW_CHECK_EQ(image1.descriptors->data.rows(), image1.keypoints->size());
    THROW_CHECK_EQ(image2.descriptors->data.rows(), image2.keypoints->size());
  }
}

FixedDimensionFeatureLocations FeatureLocations(
    const FeatureKeypoints& keypoints) {
  FixedDimensionFeatureLocations locations(keypoints.size(), 2);
  for (size_t index = 0; index < keypoints.size(); ++index) {
    locations(index, 0) = keypoints[index].x;
    locations(index, 1) = keypoints[index].y;
  }
  return locations;
}

FixedDimensionFeatureLocations NormalizedFeatureLocations(
    const Camera& camera, const FeatureKeypoints& keypoints) {
  FixedDimensionFeatureLocations locations(keypoints.size(), 2);
  for (size_t index = 0; index < keypoints.size(); ++index) {
    const auto camera_point = camera.CamFromImg(
        Eigen::Vector2d(keypoints[index].x, keypoints[index].y));
    if (camera_point.has_value()) {
      locations.row(index) = camera_point->cast<float>().transpose();
    } else {
      locations.row(index).setConstant(1e6f);
    }
  }
  return locations;
}

double NormalizedGuidedMatchingMaxResidual(const Camera& camera1,
                                           const Camera& camera2,
                                           const double max_error) {
  const double max_error1 = camera1.CamFromImgThreshold(max_error);
  const double max_error2 = camera2.CamFromImgThreshold(max_error);
  return 0.5 * (max_error1 * max_error1 + max_error2 * max_error2);
}

bool PassesGuidance(const FixedDimensionGuidanceType guidance_type,
                    const Eigen::Matrix3f& matrix,
                    const float max_residual,
                    const Eigen::Vector2f& point1,
                    const Eigen::Vector2f& point2) {
  if (guidance_type == FixedDimensionGuidanceType::EPIPOLAR) {
    const Eigen::Vector3f homogeneous1(point1.x(), point1.y(), 1.0f);
    const Eigen::Vector3f homogeneous2(point2.x(), point2.y(), 1.0f);
    const Eigen::Vector3f line2 = matrix * homogeneous1;
    const Eigen::Vector3f line1 = matrix.transpose() * homogeneous2;
    const float numerator = homogeneous2.dot(line2);
    const float denominator =
        line1.head<2>().squaredNorm() + line2.head<2>().squaredNorm();
    return numerator * numerator <= max_residual * denominator;
  }

  const Eigen::Vector3f projected =
      matrix * Eigen::Vector3f(point1.x(), point1.y(), 1.0f);
  return (projected.hnormalized() - point2).squaredNorm() <= max_residual;
}

float DescriptorDotProduct(const float* descriptor1,
                           const float* descriptor2,
                           const int descriptor_dimension) {
  float dot_product = 0.0f;
  for (int dimension = 0; dimension < descriptor_dimension; ++dimension) {
    dot_product += descriptor1[dimension] * descriptor2[dimension];
  }
  return dot_product;
}

template <typename PairFilter>
std::vector<int> FindBestMatchesOneWay(
    const FeatureDescriptorsFloatData& descriptors1,
    const FeatureDescriptorsFloatData& descriptors2,
    const bool reverse,
    const FixedDimensionMatchingOptions& options,
    PairFilter&& pair_filter) {
  const FeatureDescriptorsFloatData& query =
      reverse ? descriptors2 : descriptors1;
  const FeatureDescriptorsFloatData& reference =
      reverse ? descriptors1 : descriptors2;
  const int descriptor_dimension = query.cols();
  std::vector<int> matches(query.rows(), -1);

  for (Eigen::Index query_index = 0; query_index < query.rows();
       ++query_index) {
    int best_index = -1;
    float best_dot_product = -std::numeric_limits<float>::infinity();
    float second_dot_product = -std::numeric_limits<float>::infinity();
    for (Eigen::Index reference_index = 0; reference_index < reference.rows();
         ++reference_index) {
      const Eigen::Index index1 = reverse ? reference_index : query_index;
      const Eigen::Index index2 = reverse ? query_index : reference_index;
      if (!pair_filter(index1, index2)) {
        continue;
      }
      const float dot_product =
          DescriptorDotProduct(query.row(query_index).data(),
                               reference.row(reference_index).data(),
                               descriptor_dimension);
      if (dot_product > best_dot_product) {
        best_index = reference_index;
        second_dot_product = best_dot_product;
        best_dot_product = dot_product;
      } else if (dot_product > second_dot_product) {
        second_dot_product = dot_product;
      }
    }

    if (best_index == -1) {
      continue;
    }
    const float best_distance =
        std::acos(std::clamp(best_dot_product, -1.0f, 1.0f));
    const float second_distance =
        std::acos(std::clamp(second_dot_product, -1.0f, 1.0f));
    if (best_distance < options.max_distance &&
        best_distance < options.max_ratio * second_distance) {
      matches[query_index] = best_index;
    }
  }
  return matches;
}

template <typename PairFilter>
void FindBestMatches(const FeatureDescriptorsFloatData& descriptors1,
                     const FeatureDescriptorsFloatData& descriptors2,
                     const FixedDimensionMatchingOptions& options,
                     const int max_num_matches,
                     PairFilter&& pair_filter,
                     FeatureMatches* matches) {
  matches->clear();
  const std::vector<int> matches_1to2 = FindBestMatchesOneWay(
      descriptors1, descriptors2, false, options, pair_filter);
  const std::vector<int> matches_2to1 =
      options.cross_check
          ? FindBestMatchesOneWay(
                descriptors1, descriptors2, true, options, pair_filter)
          : std::vector<int>();

  matches->reserve(std::min<size_t>(matches_1to2.size(), max_num_matches));
  for (size_t index1 = 0; index1 < matches_1to2.size(); ++index1) {
    const int index2 = matches_1to2[index1];
    if (index2 != -1 && (!options.cross_check ||
                         matches_2to1[index2] == static_cast<int>(index1))) {
      matches->emplace_back(index1, index2);
      if (matches->size() == static_cast<size_t>(max_num_matches)) {
        break;
      }
    }
  }
}

class FixedDimensionCPUFeatureMatcher : public FeatureMatcher {
 public:
  explicit FixedDimensionCPUFeatureMatcher(
      const FeatureMatchingOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    ThrowCheckFixedDimensionFeatures(image1, image2);
    const FeatureDescriptorsFloat descriptors1 = image1.descriptors->ToFloat();
    const FeatureDescriptorsFloat descriptors2 = image2.descriptors->ToFloat();
    FindBestMatches(
        descriptors1.data,
        descriptors2.data,
        *options_.fixed_dimension,
        options_.max_num_matches,
        [](const Eigen::Index, const Eigen::Index) { return true; },
        matches);
  }

  void MatchGuided(const double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    THROW_CHECK_NOTNULL(two_view_geometry);
    ThrowCheckFixedDimensionFeatures(image1, image2, true);
    two_view_geometry->inlier_matches.clear();

    const bool use_essential_matrix =
        (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
         two_view_geometry->config == TwoViewGeometry::CALIBRATED_RIG) &&
        two_view_geometry->E.has_value();
    const bool use_fundamental_matrix =
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED &&
        two_view_geometry->F.has_value();
    const bool use_homography =
        (two_view_geometry->config == TwoViewGeometry::PLANAR ||
         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
         two_view_geometry->config == TwoViewGeometry::PLANAR_OR_PANORAMIC) &&
        two_view_geometry->H.has_value();
    if (!use_essential_matrix && !use_fundamental_matrix && !use_homography) {
      return;
    }

    const FixedDimensionGuidanceType guidance_type =
        use_homography ? FixedDimensionGuidanceType::HOMOGRAPHY
                       : FixedDimensionGuidanceType::EPIPOLAR;
    const Eigen::Matrix3f matrix =
        use_essential_matrix     ? two_view_geometry->E->cast<float>()
        : use_fundamental_matrix ? two_view_geometry->F->cast<float>()
                                 : two_view_geometry->H->cast<float>();
    const float max_residual =
        use_essential_matrix
            ? static_cast<float>(NormalizedGuidedMatchingMaxResidual(
                  *image1.camera, *image2.camera, max_error))
            : static_cast<float>(max_error * max_error);
    const FixedDimensionFeatureLocations locations1 =
        use_essential_matrix
            ? NormalizedFeatureLocations(*image1.camera, *image1.keypoints)
            : FeatureLocations(*image1.keypoints);
    const FixedDimensionFeatureLocations locations2 =
        use_essential_matrix
            ? NormalizedFeatureLocations(*image2.camera, *image2.keypoints)
            : FeatureLocations(*image2.keypoints);
    const FeatureDescriptorsFloat descriptors1 = image1.descriptors->ToFloat();
    const FeatureDescriptorsFloat descriptors2 = image2.descriptors->ToFloat();

    FindBestMatches(
        descriptors1.data,
        descriptors2.data,
        *options_.fixed_dimension,
        options_.max_num_matches,
        [&](const Eigen::Index index1, const Eigen::Index index2) {
          return PassesGuidance(
              guidance_type,
              matrix,
              max_residual,
              Eigen::Vector2f(locations1(index1, 0), locations1(index1, 1)),
              Eigen::Vector2f(locations2(index2, 0), locations2(index2, 1)));
        },
        &two_view_geometry->inlier_matches);
  }

 private:
  const FeatureMatchingOptions options_;
};

#if defined(COLMAP_CUDA_ENABLED)
class FixedDimensionGPUFeatureMatcher : public FeatureMatcher {
 public:
  explicit FixedDimensionGPUFeatureMatcher(
      const FeatureMatchingOptions& options)
      : options_(options), matcher_(options.max_num_matches) {
    THROW_CHECK(options_.Check());
    descriptor_image_ids_.fill(kInvalidImageId);
    location_image_ids_.fill(kInvalidImageId);
    normalized_locations_.fill(false);
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    ThrowCheckFixedDimensionFeatures(image1, image2);
    SetDescriptors(0, image1);
    SetDescriptors(1, image2);
    matcher_.Match(options_.fixed_dimension->max_ratio,
                   options_.fixed_dimension->max_distance,
                   options_.fixed_dimension->cross_check,
                   matches);
  }

  void MatchGuided(const double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    THROW_CHECK_NOTNULL(two_view_geometry);
    ThrowCheckFixedDimensionFeatures(image1, image2, true);
    two_view_geometry->inlier_matches.clear();

    const bool use_essential_matrix =
        (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
         two_view_geometry->config == TwoViewGeometry::CALIBRATED_RIG) &&
        two_view_geometry->E.has_value();
    const bool use_fundamental_matrix =
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED &&
        two_view_geometry->F.has_value();
    const bool use_homography =
        (two_view_geometry->config == TwoViewGeometry::PLANAR ||
         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
         two_view_geometry->config == TwoViewGeometry::PLANAR_OR_PANORAMIC) &&
        two_view_geometry->H.has_value();
    if (!use_essential_matrix && !use_fundamental_matrix && !use_homography) {
      return;
    }

    SetDescriptors(0, image1);
    SetDescriptors(1, image2);
    SetFeatureLocations(0, image1, use_essential_matrix);
    SetFeatureLocations(1, image2, use_essential_matrix);

    const FixedDimensionGuidanceType guidance_type =
        use_homography ? FixedDimensionGuidanceType::HOMOGRAPHY
                       : FixedDimensionGuidanceType::EPIPOLAR;
    const Eigen::Matrix3f matrix =
        use_essential_matrix     ? two_view_geometry->E->cast<float>()
        : use_fundamental_matrix ? two_view_geometry->F->cast<float>()
                                 : two_view_geometry->H->cast<float>();
    const float max_residual =
        use_essential_matrix
            ? static_cast<float>(NormalizedGuidedMatchingMaxResidual(
                  *image1.camera, *image2.camera, max_error))
            : static_cast<float>(max_error * max_error);
    matcher_.MatchGuided(options_.fixed_dimension->max_ratio,
                         options_.fixed_dimension->max_distance,
                         options_.fixed_dimension->cross_check,
                         guidance_type,
                         matrix,
                         max_residual,
                         &two_view_geometry->inlier_matches);
  }

 private:
  void SetDescriptors(const int image_index, const Image& image) {
    if (image.image_id != kInvalidImageId &&
        descriptor_image_ids_[image_index] == image.image_id) {
      return;
    }
    matcher_.SetDescriptors(image_index, image.descriptors->data);
    descriptor_image_ids_[image_index] = image.image_id;
  }

  void SetFeatureLocations(const int image_index,
                           const Image& image,
                           const bool normalize) {
    if (image.image_id != kInvalidImageId &&
        location_image_ids_[image_index] == image.image_id &&
        normalized_locations_[image_index] == normalize) {
      return;
    }
    matcher_.SetFeatureLocations(
        image_index,
        normalize ? NormalizedFeatureLocations(*image.camera, *image.keypoints)
                  : FeatureLocations(*image.keypoints));
    location_image_ids_[image_index] = image.image_id;
    normalized_locations_[image_index] = normalize;
  }

  const FeatureMatchingOptions options_;
  FixedDimensionCudaMatcher matcher_;
  std::array<image_t, 2> descriptor_image_ids_;
  std::array<image_t, 2> location_image_ids_;
  std::array<bool, 2> normalized_locations_;
};
#endif

}  // namespace

bool FixedDimensionMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_ratio, 0.0);
  CHECK_OPTION_GT(max_distance, 0.0);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateFixedDimensionFeatureMatcher(
    const FeatureMatchingOptions& options) {
  THROW_CHECK_EQ(options.type, FeatureMatcherType::FIXED_DIMENSION_BRUTEFORCE);
  if (options.use_gpu) {
#if defined(COLMAP_CUDA_ENABLED)
    const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1)
        << "Fixed-dimension matching can only use one GPU";
    SetBestCudaDevice(gpu_indices[0]);
    return std::make_unique<FixedDimensionGPUFeatureMatcher>(options);
#else
    return nullptr;
#endif
  }
  return std::make_unique<FixedDimensionCPUFeatureMatcher>(options);
}

}  // namespace colmap
