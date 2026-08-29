#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/geometry/rigid3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace colmap {

struct RigidBayBundleAdjustmentSummary : CasparBundleAdjustmentSummary {
  RigidBayBundleAdjustmentSummary() = default;
  explicit RigidBayBundleAdjustmentSummary(
      CasparBundleAdjustmentSummary&& summary)
      : CasparBundleAdjustmentSummary(std::move(summary)) {}

  double final_reprojection_score = 0.0;
  double final_sensor_position_prior_score = 0.0;
  double final_scale_prior_score = 0.0;
};

struct RigidBayBundleAdjustmentResult {
  std::vector<Rigid3d> bays_from_world;
  std::vector<float> points;
  double scale = 1.0;
  std::shared_ptr<RigidBayBundleAdjustmentSummary> summary;
};

RigidBayBundleAdjustmentResult RigidBayBundleAdjustmentCaspar(
    const std::vector<Rigid3d>& initial_bays_from_world,
    const float* initial_points,
    size_t num_points,
    const uint32_t* sensor_bay_indices,
    const std::vector<Rigid3d>& cameras_from_bay,
    const float* sensor_calibrations,
    const uint32_t* observation_sensor_indices,
    const uint32_t* observation_point_indices,
    const float* observation_xy,
    size_t num_observations,
    const uint32_t* prior_sensor_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    size_t num_priors,
    double initial_scale,
    double scale_prior_sqrt_information,
    const CasparBundleAdjustmentOptions& options);

}  // namespace colmap
