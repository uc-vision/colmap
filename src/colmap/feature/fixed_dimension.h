// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include "colmap/feature/matcher.h"

#include <Eigen/Core>

namespace colmap {

// Fixed-dimension descriptors are signed, L2-normalized IEEE float16 vectors
// packed row-major into FeatureDescriptorsData. Their logical dimension is
// half the byte-column count and must agree between both images. The matcher
// computes float32 cosine scores and applies angular thresholds.
using FixedDimensionFeatureLocations =
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;

enum class FixedDimensionGuidanceType {
  EPIPOLAR,
  HOMOGRAPHY,
};

struct FixedDimensionMatchingOptions {
  // Maximum distance ratio between first and second best match.
  double max_ratio = 0.8;

  // Maximum angular distance to the best match.
  double max_distance = 0.7;

  // Whether to enable cross checking in matching.
  bool cross_check = true;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateFixedDimensionFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap
