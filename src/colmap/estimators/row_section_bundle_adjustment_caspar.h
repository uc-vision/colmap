#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/row_track_caspar.h"
#include "colmap/geometry/rigid3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

struct CasparRowSectionStats {
  size_t point_count = 0;
  size_t observation_count = 0;
  size_t active_observation_count = 0;
  bool quota_truncated = false;
};

struct CasparRowSectionPlanStats {
  uint32_t tracks_per_spatial_cell = 0;
  size_t upper_bytes = 0;
};

struct CasparRowSectionResult {
  std::vector<Rigid3d> rigs_from_world;
  size_t point_count = 0;
  size_t observation_count = 0;
  size_t active_observation_count = 0;
  double preparation_seconds = 0.0;
  double optimization_seconds = 0.0;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

std::vector<CasparRowSectionStats> ComputeRowSectionStats(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint16_t* source_support,
    size_t num_row_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    size_t minimum_track_length,
    const uint32_t* density_tiers,
    size_t num_density_tiers);

CasparRowSectionPlanStats ComputeRowSectionsPlanStats(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    size_t num_row_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_masks,
    size_t num_sections,
    size_t num_frames,
    size_t minimum_track_length,
    size_t budget_bytes,
    size_t sensor_count);

CasparRowSectionResult RefineRowSectionCaspar(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint16_t* source_support,
    float* row_points,
    size_t num_row_points,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    size_t minimum_track_length,
    uint32_t tracks_per_spatial_cell,
    const CasparBundleAdjustmentOptions& options);

}  // namespace colmap
