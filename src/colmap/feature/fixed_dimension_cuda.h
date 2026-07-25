// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include "colmap/feature/fixed_dimension.h"

#include <memory>

namespace colmap {

class FixedDimensionCudaMatcher {
 public:
  explicit FixedDimensionCudaMatcher(int max_num_features);
  ~FixedDimensionCudaMatcher();

  FixedDimensionCudaMatcher(const FixedDimensionCudaMatcher&) = delete;
  FixedDimensionCudaMatcher& operator=(const FixedDimensionCudaMatcher&) =
      delete;

  void SetDescriptors(int image_index,
                      const FeatureDescriptorsData& descriptors);
  void SetFeatureLocations(int image_index,
                           const FixedDimensionFeatureLocations& locations);

  void Match(double max_ratio,
             double max_distance,
             bool cross_check,
             FeatureMatches* matches);

  void MatchGuided(double max_ratio,
                   double max_distance,
                   bool cross_check,
                   FixedDimensionGuidanceType guidance_type,
                   const Eigen::Matrix3f& guidance,
                   float max_residual,
                   FeatureMatches* matches);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap
