#include "colmap/estimators/row_track_caspar.h"

#include <Eigen/Core>

namespace colmap {

std::vector<uint32_t> RowSourceTrackOffsets(
    const std::vector<CasparRowTrackSource>& sources) {
  std::vector<uint32_t> offsets(sources.size() + 1);
  for (size_t source = 0; source < sources.size(); ++source) {
    offsets[source + 1] =
        offsets[source] + static_cast<uint32_t>(sources[source].num_tracks);
  }
  return offsets;
}

std::vector<uint32_t> SelectRowPointsByFrame(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const bool* active_frame_mask,
    const size_t num_row_points) {
  size_t candidate_track_count = 0;
  for (const CasparRowTrackSource& source : sources) {
    for (size_t image = 0; image < source.num_images; ++image) {
      if (active_frame_mask[image_frame_indices[source.image_rows[image]]]) {
        candidate_track_count += source.num_tracks;
        break;
      }
    }
  }
  std::vector<uint32_t> selected;
  selected.reserve(candidate_track_count);
  std::vector<uint8_t> selected_point(num_row_points, uint8_t{0});
  for (const CasparRowTrackSource& source : sources) {
    bool source_is_candidate = false;
    for (size_t image = 0; image < source.num_images; ++image) {
      if (active_frame_mask[image_frame_indices[source.image_rows[image]]]) {
        source_is_candidate = true;
        break;
      }
    }
    if (!source_is_candidate) {
      continue;
    }
    for (uint32_t track = 0; track < source.num_tracks; ++track) {
      bool observed_active_frame = false;
      ForEachRowTrackObservation(
          source, track, [&](const uint32_t image, const float*) {
            observed_active_frame |=
                active_frame_mask[image_frame_indices[image]];
          });
      const uint32_t row_point = source.row_point_indices[track];
      if (observed_active_frame && !selected_point[row_point]) {
        selected_point[row_point] = 1;
        selected.push_back(row_point);
      }
    }
  }
  return selected;
}

CasparRowPointSelection InitializeRowPoints(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* row_point_indices,
    const size_t num_points) {
  CasparRowPointSelection selected{
      std::vector<uint32_t>(row_point_indices, row_point_indices + num_points),
      std::vector<float>(3 * num_points),
  };
  for (size_t point = 0; point < num_points; ++point) {
    const uint32_t row_point = row_point_indices[point];
    const uint32_t point_track_start = point_track_offsets[row_point];
    const uint32_t point_track_end = point_track_offsets[row_point + 1];
    Eigen::Vector3f point_sum = Eigen::Vector3f::Zero();
    ForEachRowPointTrack(sources,
                         source_track_offsets,
                         point_track_offsets,
                         point_track_indices,
                         row_point,
                         [&](const size_t,
                             const CasparRowTrackSource& source,
                             const uint32_t track) {
                           point_sum += RowSourcePoint(
                               source, source.source_point_indices[track]);
                         });
    Eigen::Map<Eigen::Vector3f>(selected.points.data() + 3 * point) =
        point_sum / static_cast<float>(point_track_end - point_track_start);
  }
  return selected;
}

}  // namespace colmap
