#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

struct CasparPointRefinementOptions {
  int solver_iter_max = 50;
  int pcg_iter_max = 20;
  int gpu_index = -1;
  double loss_scale = 5.0;
};

struct CasparPointRefinementResult {
  std::vector<float> points;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

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

}  // namespace colmap
