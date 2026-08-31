#include "colmap/estimators/section_bundle_adjustment_arrays_caspar.h"

#include "colmap/estimators/bundle_adjustment_arrays_caspar.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <utility>

namespace colmap {
namespace {

constexpr FactorVariant kActiveVariant =
    FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;
constexpr FactorVariant kFixedVariant =
    FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;

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
  std::vector<bool> fixed_frames;
  std::vector<unsigned int> active_pose_indices;
  std::vector<size_t> active_frame_indices;
  ModelData factors;
};

SectionProblem BuildProblem(const std::vector<Rigid3d>& initial_rigs_from_world,
                            const float* initial_points,
                            const size_t num_points,
                            const uint32_t* image_frame_indices,
                            const uint32_t* image_sensor_indices,
                            const std::vector<Rigid3d>& sensors_from_rig,
                            const Sim3d& normalized_from_metric,
                            const float* sensor_calibrations,
                            const uint32_t* observation_image_indices,
                            const uint32_t* observation_point_indices,
                            const float* observation_xy,
                            const size_t num_observations,
                            const uint32_t* fixed_frame_indices,
                            const size_t num_fixed_frames) {
  SectionProblem problem{
      bundle_adjustment_arrays::NormalizeProblem(normalized_from_metric,
                                                 initial_rigs_from_world,
                                                 initial_points,
                                                 num_points,
                                                 sensors_from_rig),
      std::vector<bool>(initial_rigs_from_world.size(), false),
      std::vector<unsigned int>(initial_rigs_from_world.size()),
      {},
      {},
  };
  for (size_t index = 0; index < num_fixed_frames; ++index) {
    problem.fixed_frames[fixed_frame_indices[index]] = true;
  }
  std::vector<bool> observed_frames(initial_rigs_from_world.size(), false);
  for (size_t observation = 0; observation < num_observations; ++observation) {
    const size_t image = observation_image_indices[observation];
    observed_frames[image_frame_indices[image]] = true;
  }
  for (size_t frame = 0; frame < initial_rigs_from_world.size(); ++frame) {
    if (observed_frames[frame] && !problem.fixed_frames[frame]) {
      problem.active_pose_indices[frame] = problem.active_frame_indices.size();
      problem.active_frame_indices.push_back(frame);
    }
  }

  size_t num_fixed_observations = 0;
  for (size_t observation = 0; observation < num_observations; ++observation) {
    const size_t image = observation_image_indices[observation];
    num_fixed_observations +=
        problem.fixed_frames[image_frame_indices[image]] ? 1 : 0;
  }
  VariantData& active =
      problem.factors.variants[static_cast<int>(kActiveVariant)];
  VariantData& fixed =
      problem.factors.variants[static_cast<int>(kFixedVariant)];
  ReserveFactors(active,
                 num_observations - num_fixed_observations,
                 /*fixed_pose=*/false);
  ReserveFactors(fixed, num_fixed_observations, /*fixed_pose=*/true);

  for (size_t observation = 0; observation < num_observations; ++observation) {
    const size_t image = observation_image_indices[observation];
    const size_t frame = image_frame_indices[image];
    const size_t sensor = image_sensor_indices[image];
    if (problem.fixed_frames[frame]) {
      AppendObservation(fixed,
                        problem.normalized.sensors_from_rig[sensor],
                        sensor_calibrations + 4 * sensor,
                        observation_point_indices[observation],
                        observation_xy + 2 * observation);
      AppendPose(fixed.const_poses, problem.normalized.rigs_from_world[frame]);
    } else {
      AppendObservation(active,
                        problem.normalized.sensors_from_rig[sensor],
                        sensor_calibrations + 4 * sensor,
                        observation_point_indices[observation],
                        observation_xy + 2 * observation);
      active.pose_indices.push_back(problem.active_pose_indices[frame]);
    }
  }
  active.num_factors = active.point_indices.size();
  fixed.num_factors = fixed.point_indices.size();
  return problem;
}

}  // namespace

SectionBundleAdjustmentArraysResult SectionBundleAdjustmentArraysCaspar(
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const float* initial_points,
    const size_t num_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const std::vector<Rigid3d>& sensors_from_rig,
    const Sim3d& normalized_from_metric,
    const float* sensor_calibrations,
    const uint32_t* observation_image_indices,
    const uint32_t* observation_point_indices,
    const float* observation_xy,
    const size_t num_observations,
    const uint32_t* fixed_frame_indices,
    const size_t num_fixed_frames,
    const CasparBundleAdjustmentOptions& options) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar section array BA requires float precision";
  return {};
#else
  SectionProblem problem = BuildProblem(initial_rigs_from_world,
                                        initial_points,
                                        num_points,
                                        image_frame_indices,
                                        image_sensor_indices,
                                        sensors_from_rig,
                                        normalized_from_metric,
                                        sensor_calibrations,
                                        observation_image_indices,
                                        observation_point_indices,
                                        observation_xy,
                                        num_observations,
                                        fixed_frame_indices,
                                        num_fixed_frames);
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
  sizing.num_points = num_points;
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
  solver.SetPointNodesFromStackedHost(point_data.data(), 0, num_points);
  adapter.SetVariantFactors(solver, kActiveVariant, active);
  adapter.SetVariantFactors(solver, kFixedVariant, fixed);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve_rig_schur(
      /*print_progress=*/false,
      /*verbose_logging=*/options.collect_iteration_data);
  adapter.GetPoseNodes(solver, rig_data.data(), active_rigs.size());
  solver.GetPointNodesToStackedHost(point_data.data(), 0, num_points);

  active_rigs = bundle_adjustment_arrays::UnpackPoses(rig_data);
  const Sim3d metric_from_normalized =
      Inverse(problem.normalized.normalized_from_metric);
  bundle_adjustment_arrays::TransformPoses(active_rigs, metric_from_normalized);
  bundle_adjustment_arrays::TransformPoints(point_data, metric_from_normalized);

  SectionBundleAdjustmentArraysResult result;
  result.rigs_from_world = initial_rigs_from_world;
  for (size_t index = 0; index < active_rigs.size(); ++index) {
    result.rigs_from_world[problem.active_frame_indices[index]] =
        active_rigs[index];
  }
  result.points = std::move(point_data);
  result.summary = CasparBundleAdjustmentSummary::Create(solve_result);
  result.summary->num_residuals = 2 * num_observations;
  result.summary->allocation_size = solver.get_allocation_size();
  return result;
#endif
}

}  // namespace colmap
