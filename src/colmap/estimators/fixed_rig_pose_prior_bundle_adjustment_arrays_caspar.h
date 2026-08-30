#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

struct FixedRigPosePriorBundleAdjustmentArraysResult {
  std::vector<Rigid3d> rigs_from_world;
  std::vector<float> points;
  double sensor_from_rig_scale = 1.0;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

FixedRigPosePriorBundleAdjustmentArraysResult
FixedRigPosePriorBundleAdjustmentArraysCaspar(
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const float* initial_points,
    size_t num_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const Sim3d& normalized_from_metric,
    const float* sensor_calibrations,
    const uint32_t* observation_image_indices,
    const uint32_t* observation_point_indices,
    const float* observation_xy,
    size_t num_observations,
    const uint32_t* prior_frame_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    size_t num_priors,
    const CasparBundleAdjustmentOptions& options);

}  // namespace colmap
