#pragma once

#include "colmap/estimators/caspar/row_reprojection_validation.h"
#include "colmap/estimators/point_refinement_caspar.h"
#include "colmap/estimators/row_track_caspar.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace colmap {

struct CasparRowPointRefinementOptions : CasparPointRefinementOptions {
  bool validate_reprojection = false;
};

struct CasparRowPointRefinementSource {
  const float* colors;
  std::ptrdiff_t color_row_stride;
  std::ptrdiff_t color_column_stride;
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

std::vector<float> InitializeAllRowPoints(
    const std::vector<CasparRowTrackSource>& track_sources,
    const std::vector<CasparRowPointRefinementSource>& refinement_sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    size_t num_points,
    const uint32_t* initialized_row_point_indices,
    const float* initialized_points,
    size_t num_initialized_points);

CasparRowPointRefinementResult RefineRowPointsCaspar(
    const std::vector<CasparRowTrackSource>& track_sources,
    const std::vector<CasparRowPointRefinementSource>& refinement_sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* point_observation_counts,
    size_t num_points,
    const float* image_from_world,
    size_t num_images,
    const float* initial_points,
    size_t maximum_chunk_points,
    size_t maximum_chunk_observations,
    const CasparRowPointRefinementOptions& options);

}  // namespace colmap
