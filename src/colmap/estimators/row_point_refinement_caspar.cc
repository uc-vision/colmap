#include "colmap/estimators/row_point_refinement_caspar.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace colmap {
namespace {

using Matrix4dRowMajor = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using Clock = std::chrono::steady_clock;

double ElapsedSeconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <bool IncludeObservationOffsets>
struct ObservationOffsets {};

template <>
struct ObservationOffsets<true> {
  std::vector<uint32_t> observation_offsets;
};

template <bool IncludeObservationOffsets>
struct PackedPointChunk : ObservationOffsets<IncludeObservationOffsets> {
  std::vector<float> initial_points;
  std::vector<uint32_t> observation_point_indices;
  std::vector<uint32_t> observation_image_indices;
  std::vector<float> observation_xy;
};

Eigen::Vector3f SolvedRowTrackPoint(
    const CasparRowTrackSource& tracks,
    const CasparRowPointRefinementSource& refinement,
    const uint32_t track) {
  const int64_t observation_start = tracks.observation_offsets[track];
  const int64_t observation_end = tracks.observation_offsets[track + 1];
  const int64_t associated_observation =
      observation_start + (observation_end - observation_start) / 2;
  const int32_t associated_image =
      tracks.observation_image_indices[associated_observation];
  const Eigen::Map<const Matrix4dRowMajor> solved_world_from_source_world(
      refinement.solved_world_from_source_world + associated_image * 16);
  const int64_t source_point_index = tracks.source_point_indices[track];
  Eigen::Vector4d homogeneous_point;
  homogeneous_point
      << RowSourcePoint(tracks, source_point_index).cast<double>(),
      1.0;
  return (solved_world_from_source_world * homogeneous_point)
      .head<3>()
      .cast<float>();
}

template <bool IncludeObservationOffsets>
PackedPointChunk<IncludeObservationOffsets> PackPointChunk(
    const std::vector<CasparRowTrackSource>& track_sources,
    const std::vector<CasparRowPointRefinementSource>& refinement_sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const size_t point_start,
    const size_t point_end,
    const size_t observation_count,
    const float* initial_points,
    float* row_colors) {
  PackedPointChunk<IncludeObservationOffsets> chunk;
  chunk.initial_points.resize((point_end - point_start) * 3);
  if constexpr (IncludeObservationOffsets) {
    chunk.observation_offsets.resize(point_end - point_start + 1);
  }
  chunk.observation_point_indices.resize(observation_count);
  chunk.observation_image_indices.resize(observation_count);
  chunk.observation_xy.resize(observation_count * 2);

  size_t packed_observation = 0;
  for (size_t row_point = point_start; row_point < point_end; ++row_point) {
    Eigen::Vector3f color_sum = Eigen::Vector3f::Zero();
    const uint32_t track_start = point_track_offsets[row_point];
    const uint32_t track_end = point_track_offsets[row_point + 1];
    ForEachRowPointTrack(
        track_sources,
        source_track_offsets,
        point_track_offsets,
        point_track_indices,
        row_point,
        [&](const size_t source_index,
            const CasparRowTrackSource& tracks,
            const uint32_t track) {
          const CasparRowPointRefinementSource& refinement =
              refinement_sources[source_index];
          const int64_t source_point_index = tracks.source_point_indices[track];
          color_sum += ReadStridedVector3(refinement.colors,
                                          refinement.color_row_stride,
                                          refinement.color_column_stride,
                                          source_point_index);

          ForEachRowTrackObservation(
              tracks,
              track,
              [&](const uint32_t, const uint32_t image, const float* xy) {
                chunk.observation_point_indices[packed_observation] =
                    static_cast<uint32_t>(row_point - point_start);
                chunk.observation_image_indices[packed_observation] = image;
                Eigen::Map<Eigen::Vector2f>(chunk.observation_xy.data() +
                                            packed_observation * 2) =
                    Eigen::Map<const Eigen::Vector2f>(xy);
                ++packed_observation;
              });
        });

    const float source_count = static_cast<float>(track_end - track_start);
    Eigen::Map<Eigen::Vector3f> initial_point(chunk.initial_points.data() +
                                              (row_point - point_start) * 3);
    initial_point =
        Eigen::Map<const Eigen::Vector3f>(initial_points + row_point * 3);
    Eigen::Map<Eigen::Vector3f>(row_colors + row_point * 3) =
        color_sum / source_count;
    if constexpr (IncludeObservationOffsets) {
      chunk.observation_offsets[row_point - point_start + 1] =
          static_cast<uint32_t>(packed_observation);
    }
  }
  return chunk;
}

}  // namespace

std::vector<float> InitializeAllRowPoints(
    const std::vector<CasparRowTrackSource>& track_sources,
    const std::vector<CasparRowPointRefinementSource>& refinement_sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const size_t num_points,
    const uint32_t* initialized_row_point_indices,
    const float* initialized_points,
    const size_t num_initialized_points) {
  const std::vector<uint32_t> source_track_offsets =
      RowSourceTrackOffsets(track_sources);
  std::vector<float> points(3 * num_points);
  size_t initialized_index = 0;
  for (uint32_t row_point = 0; row_point < num_points; ++row_point) {
    while (initialized_index < num_initialized_points &&
           initialized_row_point_indices[initialized_index] < row_point) {
      ++initialized_index;
    }
    Eigen::Map<Eigen::Vector3f> point_position(points.data() + 3 * row_point);
    if (initialized_index < num_initialized_points &&
        initialized_row_point_indices[initialized_index] == row_point) {
      point_position = Eigen::Map<const Eigen::Vector3f>(initialized_points +
                                                         3 * initialized_index);
      continue;
    }

    Eigen::Vector3f point_sum = Eigen::Vector3f::Zero();
    ForEachRowPointTrack(track_sources,
                         source_track_offsets,
                         point_track_offsets,
                         point_track_indices,
                         row_point,
                         [&](const size_t source_index,
                             const CasparRowTrackSource& tracks,
                             const uint32_t track) {
                           point_sum += SolvedRowTrackPoint(
                               tracks, refinement_sources[source_index], track);
                         });
    const uint32_t track_count =
        point_track_offsets[row_point + 1] - point_track_offsets[row_point];
    point_position = point_sum / static_cast<float>(track_count);
  }
  return points;
}

template <bool ValidateReprojection>
CasparRowPointRefinementResult RefineRowPointsCasparImpl(
    const std::vector<CasparRowTrackSource>& track_sources,
    const std::vector<CasparRowPointRefinementSource>& refinement_sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* point_observation_counts,
    const size_t num_points,
    const float* image_from_world,
    const size_t num_images,
    const float* initial_points,
    const size_t maximum_chunk_points,
    const size_t maximum_chunk_observations,
    const CasparRowPointRefinementOptions& options) {
  CasparRowPointRefinementResult output;
  const Clock::time_point preparation_start = Clock::now();
  const std::vector<uint32_t> source_track_offsets =
      RowSourceTrackOffsets(track_sources);
  output.points.resize(num_points * 3);
  output.colors.resize(num_points * 3);
  output.preparation_seconds = ElapsedSeconds(preparation_start);

  std::unique_ptr<CasparRowReprojectionValidator> validator;
  if constexpr (ValidateReprojection) {
    const Clock::time_point validation_start = Clock::now();
    const size_t maximum_point_observations = *std::max_element(
        point_observation_counts, point_observation_counts + num_points);
    validator = std::make_unique<CasparRowReprojectionValidator>(
        num_points,
        image_from_world,
        num_images,
        maximum_chunk_points,
        std::max(maximum_chunk_observations, maximum_point_observations),
        SelectCasparDevice(options));
    output.validation_seconds += ElapsedSeconds(validation_start);
  }

  size_t point_start = 0;
  size_t validated_observation_count = 0;
  while (point_start < num_points) {
    const Clock::time_point packing_start = Clock::now();
    size_t point_end = point_start;
    size_t observation_count = 0;
    while (point_end < num_points &&
           point_end - point_start < maximum_chunk_points) {
      const size_t next_observation_count = point_observation_counts[point_end];
      if (point_end > point_start &&
          observation_count + next_observation_count >
              maximum_chunk_observations) {
        break;
      }
      observation_count += next_observation_count;
      ++point_end;
    }

    const PackedPointChunk<ValidateReprojection> chunk =
        PackPointChunk<ValidateReprojection>(track_sources,
                                             refinement_sources,
                                             source_track_offsets,
                                             point_track_offsets,
                                             point_track_indices,
                                             point_start,
                                             point_end,
                                             observation_count,
                                             initial_points,
                                             output.colors.data());
    output.packing_seconds += ElapsedSeconds(packing_start);
    const Clock::time_point optimization_start = Clock::now();
    CasparPointRefinementResult refined;
    if constexpr (ValidateReprojection) {
      refined = RefineFixedCameraPinholePointsCasparWithReprojectionValidation(
          chunk.initial_points.data(),
          point_end - point_start,
          image_from_world,
          num_images,
          chunk.observation_point_indices.data(),
          chunk.observation_image_indices.data(),
          chunk.observation_xy.data(),
          observation_count,
          chunk.observation_offsets.data(),
          point_start,
          options,
          *validator);
      validated_observation_count += observation_count;
    } else {
      refined = RefineFixedCameraPinholePointsCaspar(
          chunk.initial_points.data(),
          point_end - point_start,
          image_from_world,
          num_images,
          chunk.observation_point_indices.data(),
          chunk.observation_image_indices.data(),
          chunk.observation_xy.data(),
          observation_count,
          options);
    }
    output.optimization_seconds +=
        ElapsedSeconds(optimization_start) - refined.validation_seconds;
    if constexpr (ValidateReprojection) {
      output.validation_seconds += refined.validation_seconds;
    }
    if (!refined.summary->IsSolutionUsable()) {
      throw std::runtime_error(refined.summary->BriefReport());
    }
    std::copy(refined.points.begin(),
              refined.points.end(),
              output.points.begin() + point_start * 3);
    ++output.chunk_count;
    output.iteration_count += refined.summary->iteration_count;
    output.maximum_iterations_run = std::max(output.maximum_iterations_run,
                                             refined.summary->iteration_count);
    output.runtime_seconds += refined.summary->runtime;
    output.initial_score += refined.summary->initial_score;
    output.final_score += refined.summary->final_score;
    point_start = point_end;
  }
  if constexpr (ValidateReprojection) {
    const Clock::time_point validation_start = Clock::now();
    output.reprojection = validator->Summarize(validated_observation_count);
    output.validation_seconds += ElapsedSeconds(validation_start);
    const Clock::time_point release_start = Clock::now();
    validator.reset();
    output.validation_seconds += ElapsedSeconds(release_start);
  }
  return output;
}

CasparRowPointRefinementResult RefineRowPointsCaspar(
    const std::vector<CasparRowTrackSource>& track_sources,
    const std::vector<CasparRowPointRefinementSource>& refinement_sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* point_observation_counts,
    const size_t num_points,
    const float* image_from_world,
    const size_t num_images,
    const float* initial_points,
    const size_t maximum_chunk_points,
    const size_t maximum_chunk_observations,
    const CasparRowPointRefinementOptions& options) {
  if (options.validate_reprojection) {
    return RefineRowPointsCasparImpl<true>(track_sources,
                                           refinement_sources,
                                           point_track_offsets,
                                           point_track_indices,
                                           point_observation_counts,
                                           num_points,
                                           image_from_world,
                                           num_images,
                                           initial_points,
                                           maximum_chunk_points,
                                           maximum_chunk_observations,
                                           options);
  }
  return RefineRowPointsCasparImpl<false>(track_sources,
                                          refinement_sources,
                                          point_track_offsets,
                                          point_track_indices,
                                          point_observation_counts,
                                          num_points,
                                          image_from_world,
                                          num_images,
                                          initial_points,
                                          maximum_chunk_points,
                                          maximum_chunk_observations,
                                          options);
}

}  // namespace colmap
