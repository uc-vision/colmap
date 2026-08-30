#pragma once

#include "colmap/estimators/caspar/row_reprojection_validation.h"
#include "colmap/estimators/point_refinement_caspar.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace colmap {

struct CasparRowPointRefinementOptions : CasparPointRefinementOptions {
  bool validate_reprojection = false;
};

struct CasparRowPointRefinementSource {
  const float* points;
  const float* colors;
  const int64_t* source_point_indices;
  size_t num_tracks;
  const int64_t* observation_offsets;
  const int32_t* observation_image_indices;
  const float* observation_xy;
  const uint32_t* duplicate_observation_indices;
  size_t num_duplicate_observations;
  const uint32_t* solved_image_rows;
  const double* solved_world_from_source_world;
};

struct CasparRowPointRefinementResult {
  std::vector<float> points;
  std::vector<float> colors;
  size_t chunk_count = 0;
  int iteration_count = 0;
  int maximum_iterations_run = 0;
  double runtime_seconds = 0.0;
  double initial_score = 0.0;
  double final_score = 0.0;
  double preparation_seconds = 0.0;
  double packing_seconds = 0.0;
  double optimization_seconds = 0.0;
  double validation_seconds = 0.0;
  CasparReprojectionErrorSummary reprojection;
};

CasparRowPointRefinementResult RefineRowPointsCaspar(
    const std::vector<CasparRowPointRefinementSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* point_observation_counts,
    size_t num_points,
    const float* image_from_world,
    size_t num_images,
    const uint32_t* initialized_row_point_indices,
    const float* initialized_points,
    size_t num_initialized_points,
    size_t maximum_chunk_points,
    size_t maximum_chunk_observations,
    const CasparRowPointRefinementOptions& options);

}  // namespace colmap
