#include "colmap/estimators/row_section_bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_arrays_caspar.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace colmap {
namespace {

using Clock = std::chrono::steady_clock;

constexpr FactorVariant kActiveVariant =
    FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;
constexpr FactorVariant kFixedVariant =
    FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;

double ElapsedSeconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

size_t SectionUpperBytes(const CasparRowSectionStats& stats,
                         const size_t frame_count,
                         const size_t sensor_count) {
  return 320 * stats.point_count + 288 * stats.active_observation_count +
         160 * (stats.observation_count - stats.active_observation_count) +
         1024 * frame_count + 92 * sensor_count + 620;
}

void ReserveFactors(VariantData& factors,
                    const size_t num_factors,
                    const bool fixed_pose) {
  factors.pose_indices.reserve(fixed_pose ? 0 : num_factors);
  factors.point_indices.reserve(num_factors);
  factors.sensor_from_rig_data.reserve(7 * num_factors);
  factors.const_poses.reserve(fixed_pose ? 7 * num_factors : 0);
  factors.const_focal_and_extra.reserve(2 * num_factors);
  factors.const_principal_point.reserve(2 * num_factors);
  factors.pixels.reserve(2 * num_factors);
}

void AppendPose(std::vector<StorageType>& data, const Rigid3d& pose) {
  const Eigen::Matrix<StorageType, 7, 1> packed =
      pose.params.cast<StorageType>();
  data.insert(data.end(), packed.data(), packed.data() + packed.size());
}

void AppendObservation(VariantData& factors,
                       const Rigid3d& sensor_from_rig,
                       const float* calibration,
                       const uint32_t point_index,
                       const float* pixel) {
  AppendPose(factors.sensor_from_rig_data, sensor_from_rig);
  factors.const_focal_and_extra.push_back(calibration[0]);
  factors.const_focal_and_extra.push_back(calibration[1]);
  factors.const_principal_point.push_back(calibration[2]);
  factors.const_principal_point.push_back(calibration[3]);
  factors.point_indices.push_back(point_index);
  factors.pixels.push_back(pixel[0]);
  factors.pixels.push_back(pixel[1]);
}

struct SectionProblem {
  bundle_adjustment_arrays::NormalizedProblem normalized;
  std::vector<unsigned int> active_pose_indices;
  std::vector<size_t> active_frame_indices;
  ModelData factors;
  size_t observation_count;
  size_t active_observation_count;
};

struct SectionObservations {
  std::vector<bool> active_images;
  std::vector<bool> active_frames;
  size_t observation_count = 0;
  size_t active_observation_count = 0;
};

SectionObservations AnalyzeObservations(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const std::vector<uint32_t>& row_point_indices,
    const uint32_t* image_frame_indices,
    const bool* active_frame_mask,
    const size_t num_frames,
    const size_t num_images) {
  SectionObservations observations{
      std::vector<bool>(num_images, false),
      std::vector<bool>(num_frames, false),
  };
  ForEachRowObservation(
      sources,
      source_track_offsets,
      point_track_offsets,
      point_track_indices,
      row_point_indices,
      [&](const uint32_t, const uint32_t image, const float*) {
        const uint32_t frame = image_frame_indices[image];
        ++observations.observation_count;
        if (active_frame_mask[frame]) {
          ++observations.active_observation_count;
          observations.active_images[image] = true;
          observations.active_frames[frame] = true;
        }
      });
  return observations;
}

SectionProblem BuildProblem(const std::vector<CasparRowTrackSource>& sources,
                            const std::vector<uint32_t>& source_track_offsets,
                            const uint32_t* point_track_offsets,
                            const uint32_t* point_track_indices,
                            const CasparRowPointSelection& selected,
                            const std::vector<Rigid3d>& initial_rigs_from_world,
                            const uint32_t* image_frame_indices,
                            const uint32_t* image_sensor_indices,
                            const size_t num_images,
                            const std::vector<Rigid3d>& sensors_from_rig,
                            const float* sensor_calibrations,
                            const bool* active_frame_mask) {
  const SectionObservations observations =
      AnalyzeObservations(sources,
                          source_track_offsets,
                          point_track_offsets,
                          point_track_indices,
                          selected.row_point_indices,
                          image_frame_indices,
                          active_frame_mask,
                          initial_rigs_from_world.size(),
                          num_images);
  const size_t fixed_observation_count =
      observations.observation_count - observations.active_observation_count;

  std::vector<uint32_t> active_images;
  for (uint32_t image = 0; image < num_images; ++image) {
    if (observations.active_images[image]) {
      active_images.push_back(image);
    }
  }
  const Sim3d normalized_from_metric =
      bundle_adjustment_arrays::ComputeNormalization(initial_rigs_from_world,
                                                     image_frame_indices,
                                                     image_sensor_indices,
                                                     active_images.data(),
                                                     active_images.size(),
                                                     sensors_from_rig);
  SectionProblem problem{
      bundle_adjustment_arrays::NormalizeProblem(
          normalized_from_metric,
          initial_rigs_from_world,
          selected.points.data(),
          selected.row_point_indices.size(),
          sensors_from_rig),
      std::vector<unsigned int>(initial_rigs_from_world.size()),
      {},
      {},
      observations.observation_count,
      observations.active_observation_count,
  };
  for (size_t frame = 0; frame < observations.active_frames.size(); ++frame) {
    if (observations.active_frames[frame]) {
      problem.active_pose_indices[frame] = problem.active_frame_indices.size();
      problem.active_frame_indices.push_back(frame);
    }
  }

  VariantData& active =
      problem.factors.variants[static_cast<int>(kActiveVariant)];
  VariantData& fixed =
      problem.factors.variants[static_cast<int>(kFixedVariant)];
  ReserveFactors(active,
                 observations.active_observation_count,
                 /*fixed_pose=*/false);
  ReserveFactors(fixed, fixed_observation_count, /*fixed_pose=*/true);
  ForEachRowObservation(
      sources,
      source_track_offsets,
      point_track_offsets,
      point_track_indices,
      selected.row_point_indices,
      [&](const uint32_t point, const uint32_t image, const float* xy) {
        const uint32_t frame = image_frame_indices[image];
        const uint32_t sensor = image_sensor_indices[image];
        if (active_frame_mask[frame]) {
          AppendObservation(active,
                            problem.normalized.sensors_from_rig[sensor],
                            sensor_calibrations + 4 * sensor,
                            point,
                            xy);
          active.pose_indices.push_back(problem.active_pose_indices[frame]);
        } else {
          AppendObservation(fixed,
                            problem.normalized.sensors_from_rig[sensor],
                            sensor_calibrations + 4 * sensor,
                            point,
                            xy);
          AppendPose(fixed.const_poses,
                     problem.normalized.rigs_from_world[frame]);
        }
      });
  active.num_factors = active.point_indices.size();
  fixed.num_factors = fixed.point_indices.size();
  return problem;
}

}  // namespace

std::vector<CasparRowSectionStats> ComputeRowSectionStats(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint16_t* source_support,
    const size_t num_row_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    const size_t minimum_track_length,
    const uint32_t* density_tiers,
    const size_t num_density_tiers) {
  const std::vector<uint32_t> source_track_offsets =
      RowSourceTrackOffsets(sources);
  const CasparRowTiers assignment = AssignCasparRowTiers(sources,
                                                         image_frame_indices,
                                                         image_sensor_indices,
                                                         sensor_dimensions,
                                                         active_frame_mask,
                                                         num_row_points,
                                                         minimum_track_length,
                                                         density_tiers,
                                                         num_density_tiers);
  std::vector<CasparRowSectionStats> stats(num_density_tiers);
  for (uint32_t row_point = 0; row_point < num_row_points; ++row_point) {
    const uint8_t tier = assignment.first_tier[row_point];
    if (tier == num_density_tiers) {
      continue;
    }
    CasparRowSectionStats& tier_stats = stats[tier];
    ++tier_stats.point_count;
    ForEachRowPointTrack(
        sources,
        source_track_offsets,
        point_track_offsets,
        point_track_indices,
        row_point,
        [&](const size_t,
            const CasparRowTrackSource& source,
            const uint32_t track) {
          ForEachRowTrackObservation(
              source,
              track,
              [&](const uint32_t, const uint32_t image, const float*) {
                ++tier_stats.observation_count;
                if (active_frame_mask[image_frame_indices[image]]) {
                  ++tier_stats.active_observation_count;
                }
              });
        });
  }
  for (size_t tier = 1; tier < stats.size(); ++tier) {
    stats[tier].point_count += stats[tier - 1].point_count;
    stats[tier].observation_count += stats[tier - 1].observation_count;
    stats[tier].active_observation_count +=
        stats[tier - 1].active_observation_count;
  }
  stats.back().quota_truncated = assignment.quota_truncated;
  return stats;
}

CasparRowSectionPlanStats ComputeRowSectionsPlanStats(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const size_t num_row_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_masks,
    const size_t num_sections,
    const size_t num_frames,
    const size_t minimum_track_length,
    const size_t budget_bytes,
    const size_t sensor_count) {
  const uint32_t unselected = std::numeric_limits<uint32_t>::max();
  CasparRowSectionDensities assignment =
      AssignCasparRowSectionDensities(sources,
                                      image_frame_indices,
                                      image_sensor_indices,
                                      sensor_dimensions,
                                      active_frame_masks,
                                      num_sections,
                                      num_frames,
                                      num_row_points,
                                      minimum_track_length);
  std::vector<uint32_t> maximum_density(num_sections);
  for (size_t section = 0; section < num_sections; ++section) {
    const uint32_t* densities =
        assignment.first_density.data() + section * num_row_points;
    for (size_t point = 0; point < num_row_points; ++point) {
      if (densities[point] != unselected) {
        maximum_density[section] =
            std::max(maximum_density[section], densities[point]);
      }
    }
  }
  std::vector<std::vector<CasparRowSectionStats>> histograms(num_sections);
  for (size_t section = 0; section < num_sections; ++section) {
    histograms[section].resize(maximum_density[section] + 1);
  }
  for (size_t point = 0; point < num_row_points; ++point) {
    for (size_t section = 0; section < num_sections; ++section) {
      const uint32_t density =
          assignment.first_density[section * num_row_points + point];
      if (density != unselected) {
        CasparRowSectionStats& stats = histograms[section][density];
        ++stats.point_count;
        stats.observation_count += assignment.observation_counts[point];
      }
    }
  }

  std::vector<uint32_t> frame_section_offsets(num_frames + 1);
  for (size_t frame = 0; frame < num_frames; ++frame) {
    for (size_t section = 0; section < num_sections; ++section) {
      frame_section_offsets[frame + 1] +=
          active_frame_masks[section * num_frames + frame];
    }
    frame_section_offsets[frame + 1] += frame_section_offsets[frame];
  }
  std::vector<uint32_t> frame_sections(frame_section_offsets.back());
  std::vector<uint32_t> writes = frame_section_offsets;
  for (uint32_t frame = 0; frame < num_frames; ++frame) {
    for (uint32_t section = 0; section < num_sections; ++section) {
      if (active_frame_masks[section * num_frames + frame]) {
        frame_sections[writes[frame]++] = section;
      }
    }
  }
  for (const CasparRowTrackSource& source : sources) {
    for (uint32_t track = 0; track < source.num_tracks; ++track) {
      const uint32_t point = source.row_point_indices[track];
      ForEachRowTrackObservation(
          source,
          track,
          [&](const uint32_t, const uint32_t image, const float*) {
            const uint32_t frame = image_frame_indices[image];
            for (uint32_t membership = frame_section_offsets[frame];
                 membership < frame_section_offsets[frame + 1];
                 ++membership) {
              const uint32_t section = frame_sections[membership];
              const uint32_t density =
                  assignment.first_density[section * num_row_points + point];
              if (density != unselected) {
                ++histograms[section][density].active_observation_count;
              }
            }
          });
    }
  }
  assignment = {};

  uint32_t largest_density = 0;
  for (const uint32_t density : maximum_density) {
    largest_density = std::max(largest_density, density);
  }
  CasparRowSectionPlanStats plan;
  for (uint32_t density = 0; density <= largest_density; ++density) {
    size_t largest_bytes = 0;
    for (size_t section = 0; section < num_sections; ++section) {
      std::vector<CasparRowSectionStats>& histogram = histograms[section];
      if (density > 0 && density < histogram.size()) {
        CasparRowSectionStats& stats = histogram[density];
        const CasparRowSectionStats& previous = histogram[density - 1];
        stats.point_count += previous.point_count;
        stats.observation_count += previous.observation_count;
        stats.active_observation_count += previous.active_observation_count;
      }
      const CasparRowSectionStats& stats =
          histogram[std::min<size_t>(density, histogram.size() - 1)];
      largest_bytes = std::max(
          largest_bytes, SectionUpperBytes(stats, num_frames, sensor_count));
    }
    if (largest_bytes > budget_bytes) {
      break;
    }
    plan.tracks_per_spatial_cell = density;
    plan.upper_bytes = largest_bytes;
  }
  return plan;
}

CasparRowSectionResult RefineRowSectionCaspar(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint16_t* source_support,
    float* row_points,
    const size_t num_row_points,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    const size_t minimum_track_length,
    const uint32_t tracks_per_spatial_cell,
    const CasparBundleAdjustmentOptions& options) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar row section BA requires float precision";
  return {};
#else
  const Clock::time_point preparation_start = Clock::now();
  const std::vector<uint32_t> source_track_offsets =
      RowSourceTrackOffsets(sources);
  CasparRowTrackSelection track_selection =
      SelectCasparRowPoints(sources,
                            image_frame_indices,
                            image_sensor_indices,
                            sensor_dimensions,
                            active_frame_mask,
                            num_row_points,
                            minimum_track_length,
                            tracks_per_spatial_cell);
  CasparRowPointSelection selected;
  selected.row_point_indices = std::move(track_selection.point_indices);
  selected.points.reserve(3 * selected.row_point_indices.size());
  for (const uint32_t row_point : selected.row_point_indices) {
    selected.points.insert(selected.points.end(),
                           row_points + 3 * row_point,
                           row_points + 3 * row_point + 3);
  }
  SectionProblem problem = BuildProblem(sources,
                                        source_track_offsets,
                                        point_track_offsets,
                                        point_track_indices,
                                        selected,
                                        initial_rigs_from_world,
                                        image_frame_indices,
                                        image_sensor_indices,
                                        num_images,
                                        sensors_from_rig,
                                        sensor_calibrations,
                                        active_frame_mask);
  const double preparation_seconds = ElapsedSeconds(preparation_start);

  const Clock::time_point optimization_start = Clock::now();
  std::vector<Rigid3d> active_rigs;
  active_rigs.reserve(problem.active_frame_indices.size());
  for (const size_t frame : problem.active_frame_indices) {
    active_rigs.push_back(problem.normalized.rigs_from_world[frame]);
  }
  std::vector<StorageType> rig_data =
      bundle_adjustment_arrays::PackPoses(active_rigs);
  std::vector<StorageType> point_data = std::move(problem.normalized.points);

  PinholeAdapter adapter;
  CasparSolverSizing sizing;
  sizing.num_pinhole_poses = active_rigs.size();
  sizing.num_points = selected.row_point_indices.size();
  adapter.FillSizing(sizing, problem.factors, /*num_calibs=*/0);
  const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  const int gpu_index = gpu_indices.front();
  const size_t device_id =
      static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
  auto solver =
      CreateSolver(CreateCasparSolverParameters(options), sizing, device_id);

  VariantData& active =
      problem.factors.variants[static_cast<int>(kActiveVariant)];
  VariantData& fixed =
      problem.factors.variants[static_cast<int>(kFixedVariant)];
  solver.SetRigSchurTopology(active.pose_indices,
                             active.point_indices,
                             fixed.point_indices,
                             std::vector<unsigned int>{});
  adapter.SetPoseNodes(solver, rig_data.data(), active_rigs.size());
  solver.SetPointNodesFromStackedHost(
      point_data.data(), 0, selected.row_point_indices.size());
  adapter.SetVariantFactors(solver, kActiveVariant, active);
  adapter.SetVariantFactors(solver, kFixedVariant, fixed);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve_rig_schur(
      /*print_progress=*/false,
      /*verbose_logging=*/options.collect_iteration_data);
  adapter.GetPoseNodes(solver, rig_data.data(), active_rigs.size());
  solver.GetPointNodesToStackedHost(
      point_data.data(), 0, selected.row_point_indices.size());

  active_rigs = bundle_adjustment_arrays::UnpackPoses(rig_data);
  const Sim3d metric_from_normalized =
      Inverse(problem.normalized.normalized_from_metric);
  bundle_adjustment_arrays::TransformPoses(active_rigs, metric_from_normalized);
  bundle_adjustment_arrays::TransformPoints(point_data, metric_from_normalized);
  for (size_t point = 0; point < selected.row_point_indices.size(); ++point) {
    std::copy_n(point_data.data() + 3 * point,
                3,
                row_points + 3 * selected.row_point_indices[point]);
  }

  CasparRowSectionResult result;
  result.rigs_from_world = initial_rigs_from_world;
  for (size_t index = 0; index < active_rigs.size(); ++index) {
    result.rigs_from_world[problem.active_frame_indices[index]] =
        active_rigs[index];
  }
  result.point_count = selected.row_point_indices.size();
  result.observation_count = problem.observation_count;
  result.active_observation_count = problem.active_observation_count;
  result.preparation_seconds = preparation_seconds;
  result.optimization_seconds = ElapsedSeconds(optimization_start);
  result.summary = CasparBundleAdjustmentSummary::Create(solve_result);
  result.summary->num_residuals = 2 * problem.observation_count;
  result.summary->allocation_size = solver.get_allocation_size();
  return result;
#endif
}

}  // namespace colmap
