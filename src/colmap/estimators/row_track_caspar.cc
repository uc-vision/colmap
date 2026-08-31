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

CasparRowPointSelection SelectRowPointsBySource(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const bool* active_source_mask,
    const uint32_t* eligible_row_point_indices,
    const float* eligible_points,
    const size_t num_eligible_points) {
  CasparRowPointSelection selected;
  selected.row_point_indices.reserve(num_eligible_points);
  selected.points.reserve(3 * num_eligible_points);
  for (size_t eligible = 0; eligible < num_eligible_points; ++eligible) {
    const uint32_t row_point = eligible_row_point_indices[eligible];
    bool selected_point = false;
    ForEachRowPointTrack(sources,
                         source_track_offsets,
                         point_track_offsets,
                         point_track_indices,
                         row_point,
                         [&](const size_t source_index,
                             const CasparRowTrackSource&,
                             const uint32_t) {
                           selected_point |= active_source_mask[source_index];
                         });
    if (!selected_point) {
      continue;
    }
    selected.row_point_indices.push_back(row_point);
    selected.points.insert(selected.points.end(),
                           eligible_points + 3 * eligible,
                           eligible_points + 3 * eligible + 3);
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
    ForEachRowPointTrack(
        sources,
        source_track_offsets,
        point_track_offsets,
        point_track_indices,
        row_point,
        [&](const size_t,
            const CasparRowTrackSource& source,
            const uint32_t track) {
          point_sum += Eigen::Map<const Eigen::Vector3f>(
              source.points + 3 * source.source_point_indices[track]);
        });
    Eigen::Map<Eigen::Vector3f>(selected.points.data() + 3 * point) =
        point_sum / static_cast<float>(point_track_end - point_track_start);
  }
  return selected;
}

}  // namespace colmap
