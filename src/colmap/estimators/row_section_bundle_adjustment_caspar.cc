#include "colmap/estimators/row_section_bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_arrays_caspar.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <chrono>
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
  size_t active_observation_count = 0;
  size_t fixed_observation_count = 0;
  std::vector<bool> observed_active_images(num_images, false);
  std::vector<bool> observed_active_frames(initial_rigs_from_world.size(),
                                           false);
  ForEachRowObservation(
      sources,
      source_track_offsets,
      point_track_offsets,
      point_track_indices,
      selected.row_point_indices,
      [&](const uint32_t, const uint32_t image, const float*) {
        const uint32_t frame = image_frame_indices[image];
        if (active_frame_mask[frame]) {
          ++active_observation_count;
          observed_active_images[image] = true;
          observed_active_frames[frame] = true;
        } else {
          ++fixed_observation_count;
        }
      });

  std::vector<uint32_t> active_images;
  for (uint32_t image = 0; image < num_images; ++image) {
    if (observed_active_images[image]) {
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
      active_observation_count + fixed_observation_count,
      active_observation_count,
  };
  for (size_t frame = 0; frame < observed_active_frames.size(); ++frame) {
    if (observed_active_frames[frame]) {
      problem.active_pose_indices[frame] = problem.active_frame_indices.size();
      problem.active_frame_indices.push_back(frame);
    }
  }

  VariantData& active =
      problem.factors.variants[static_cast<int>(kActiveVariant)];
  VariantData& fixed =
      problem.factors.variants[static_cast<int>(kFixedVariant)];
  ReserveFactors(active, active_observation_count, /*fixed_pose=*/false);
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

CasparRowSectionResult RefineRowSectionCaspar(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* eligible_row_point_indices,
    const float* eligible_points,
    const size_t num_eligible_points,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const bool* active_frame_mask,
    const bool* active_source_mask,
    const CasparBundleAdjustmentOptions& options) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar row section BA requires float precision";
  return {};
#else
  const Clock::time_point preparation_start = Clock::now();
  const std::vector<uint32_t> source_track_offsets =
      RowSourceTrackOffsets(sources);
  CasparRowPointSelection selected =
      SelectRowPointsBySource(sources,
                              source_track_offsets,
                              point_track_offsets,
                              point_track_indices,
                              active_source_mask,
                              eligible_row_point_indices,
                              eligible_points,
                              num_eligible_points);
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

  CasparRowSectionResult result;
  result.rigs_from_world = initial_rigs_from_world;
  for (size_t index = 0; index < active_rigs.size(); ++index) {
    result.rigs_from_world[problem.active_frame_indices[index]] =
        active_rigs[index];
  }
  result.row_point_indices = std::move(selected.row_point_indices);
  result.points = std::move(point_data);
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
