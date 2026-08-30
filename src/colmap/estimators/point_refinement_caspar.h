#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

class CasparRowReprojectionValidator;

struct CasparPointRefinementOptions : CasparSolverOptions {
  CasparPointRefinementOptions() { solver_iter_max = 50; }
};

struct CasparPointRefinementResult {
  std::vector<float> points;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
  double validation_seconds = 0.0;
};

size_t SelectCasparDevice(const CasparSolverOptions& options);

CasparPointRefinementResult RefineFixedCameraPinholePointsCaspar(
    const float* initial_points,
    size_t num_points,
    const float* image_from_world,
    size_t num_images,
    const uint32_t* observation_point_indices,
    const uint32_t* observation_image_indices,
    const float* observation_xy,
    size_t num_observations,
    const CasparPointRefinementOptions& options);

CasparPointRefinementResult
RefineFixedCameraPinholePointsCasparWithReprojectionValidation(
    const float* initial_points,
    size_t num_points,
    const float* image_from_world,
    size_t num_images,
    const uint32_t* observation_point_indices,
    const uint32_t* observation_image_indices,
    const float* observation_xy,
    size_t num_observations,
    const uint32_t* observation_offsets,
    size_t row_point_start,
    const CasparPointRefinementOptions& options,
    CasparRowReprojectionValidator& validator);

}  // namespace colmap
