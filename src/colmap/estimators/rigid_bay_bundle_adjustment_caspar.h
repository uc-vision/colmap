#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/geometry/rigid3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

struct RigidBayBundleAdjustmentResult {
  std::vector<Rigid3d> bays_from_world;
  std::vector<float> points;
  double scale = 1.0;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
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
    const CasparBundleAdjustmentOptions& options);

}  // namespace colmap
