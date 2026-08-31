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

CasparRowSectionStats ComputeRowSectionStats(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    size_t num_row_points,
    const uint32_t* image_frame_indices,
    const bool* active_frame_mask,
    size_t num_frames,
    size_t num_images);

CasparRowSectionResult RefineRowSectionCaspar(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    float* row_points,
    size_t num_row_points,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const bool* active_frame_mask,
    const CasparBundleAdjustmentOptions& options);

}  // namespace colmap
