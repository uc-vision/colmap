#include "colmap/estimators/rigid_bay_bundle_adjustment_caspar.h"

#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Core>

namespace colmap {
namespace {

using PackedPose = Eigen::Matrix<StorageType, 7, 1>;
using PackedLogScale = Eigen::Matrix<StorageType, 1, 1>;
using InputSqrtInformation = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using PackedSqrtInformation = Eigen::Matrix<StorageType, 3, 3>;

struct ObservationFactorData {
  std::vector<unsigned int> pose_indices;
  std::vector<unsigned int> point_indices;
  std::vector<StorageType> sensors_from_bay;
  std::vector<StorageType> calibrations;
};

struct PriorFactorData {
  std::vector<unsigned int> pose_indices;
  std::vector<StorageType> sensors_from_bay;
  std::vector<StorageType> positions;
  std::vector<StorageType> sqrt_information;
};

std::vector<StorageType> PackPoses(const std::vector<Rigid3d>& poses) {
  std::vector<StorageType> packed(7 * poses.size());
  for (size_t index = 0; index < poses.size(); ++index) {
    Eigen::Map<PackedPose>(packed.data() + 7 * index) =
        poses[index].params.cast<StorageType>();
  }
  return packed;
}

std::vector<Rigid3d> UnpackPoses(const std::vector<StorageType>& packed) {
  std::vector<Rigid3d> poses(packed.size() / 7);
  for (size_t index = 0; index < poses.size(); ++index) {
    poses[index].params =
        Eigen::Map<const PackedPose>(packed.data() + 7 * index).cast<double>();
    poses[index].rotation().normalize();
  }
  return poses;
}

ObservationFactorData BuildObservationFactorData(
    const uint32_t* sensor_bay_indices,
    const std::vector<StorageType>& packed_sensors,
    const float* sensor_calibrations,
    const uint32_t* observation_sensor_indices,
    const uint32_t* observation_point_indices,
    const size_t num_observations) {
  ObservationFactorData data{
      std::vector<unsigned int>(num_observations),
      std::vector<unsigned int>(observation_point_indices,
                                observation_point_indices + num_observations),
      std::vector<StorageType>(7 * num_observations),
      std::vector<StorageType>(4 * num_observations)};
  for (size_t observation = 0; observation < num_observations; ++observation) {
    const size_t sensor = observation_sensor_indices[observation];
    data.pose_indices[observation] = sensor_bay_indices[sensor];
    std::copy_n(packed_sensors.data() + 7 * sensor,
                7,
                data.sensors_from_bay.data() + 7 * observation);
    std::copy_n(sensor_calibrations + 4 * sensor,
                4,
                data.calibrations.data() + 4 * observation);
  }
  return data;
}

PriorFactorData BuildPriorFactorData(
    const uint32_t* sensor_bay_indices,
    const std::vector<StorageType>& packed_sensors,
    const uint32_t* prior_sensor_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    const size_t num_priors) {
  PriorFactorData data{std::vector<unsigned int>(num_priors),
                       std::vector<StorageType>(7 * num_priors),
                       std::vector<StorageType>(3 * num_priors),
                       std::vector<StorageType>(9 * num_priors)};
  for (size_t prior = 0; prior < num_priors; ++prior) {
    const size_t sensor = prior_sensor_indices[prior];
    data.pose_indices[prior] = sensor_bay_indices[sensor];
    std::copy_n(packed_sensors.data() + 7 * sensor,
                7,
                data.sensors_from_bay.data() + 7 * prior);
    Eigen::Map<Eigen::Matrix<StorageType, 3, 1>>(
        data.positions.data() + 3 * prior) =
        Eigen::Map<const Eigen::Vector3f>(prior_positions + 3 * prior)
            .cast<StorageType>();
    Eigen::Map<PackedSqrtInformation>(
        data.sqrt_information.data() + 9 * prior) =
        Eigen::Map<const InputSqrtInformation>(prior_sqrt_information +
                                               9 * prior)
            .cast<StorageType>();
  }
  return data;
}

double ComputeSensorPositionPriorScore(
    const std::vector<Rigid3d>& bays_from_world,
    const uint32_t* sensor_bay_indices,
    const std::vector<Rigid3d>& cameras_from_bay,
    const uint32_t* prior_sensor_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    const size_t num_priors,
    const double scale) {
  double score = 0.0;
  for (size_t prior = 0; prior < num_priors; ++prior) {
    const size_t sensor = prior_sensor_indices[prior];
    Rigid3d scaled_camera_from_bay = cameras_from_bay[sensor];
    scaled_camera_from_bay.translation() *= scale;
    const Rigid3d camera_from_world =
        scaled_camera_from_bay * bays_from_world[sensor_bay_indices[sensor]];
    const Eigen::Vector3d residual =
        Eigen::Map<const InputSqrtInformation>(prior_sqrt_information +
                                               9 * prior)
            .cast<double>() *
        (camera_from_world.TgtOriginInSrc() -
         Eigen::Map<const Eigen::Vector3f>(prior_positions + 3 * prior)
             .cast<double>());
    score += residual.squaredNorm();
  }
  return 0.5 * score;
}

caspar::SolverParams<double> CreateSolverParameters(
    const CasparBundleAdjustmentOptions& options) {
  caspar::SolverParams<double> parameters;
  parameters.solver_iter_max = options.solver_iter_max;
  parameters.pcg_iter_max = options.pcg_iter_max;
  parameters.diag_init = options.diag_init;
  parameters.diag_min = options.diag_min;
  parameters.diag_scaling_up = options.diag_scaling_up;
  parameters.diag_scaling_down = options.diag_scaling_down;
  parameters.diag_exit_value = options.diag_exit_value;
  parameters.score_exit_value = options.score_exit_value;
  parameters.pcg_rel_error_exit = options.pcg_rel_error_exit;
  parameters.pcg_rel_score_exit = options.pcg_rel_score_exit;
  parameters.pcg_rel_decrease_min = options.pcg_rel_decrease_min;
  parameters.solver_rel_decrease_min = options.solver_rel_decrease_min;
  return parameters;
}

}  // namespace

RigidBayBundleAdjustmentResult RigidBayBundleAdjustmentCaspar(
    const std::vector<Rigid3d>& initial_bays_from_world,
    const float* initial_points,
    const size_t num_points,
    const uint32_t* sensor_bay_indices,
    const std::vector<Rigid3d>& cameras_from_bay,
    const float* sensor_calibrations,
    const uint32_t* observation_sensor_indices,
    const uint32_t* observation_point_indices,
    const float* observation_xy,
    const size_t num_observations,
    const uint32_t* prior_sensor_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    const size_t num_priors,
    const double initial_scale,
    const double scale_prior_sqrt_information,
    const CasparBundleAdjustmentOptions& options) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar rigid-bay BA requires float precision";
  return {};
#else
  CasparSolverSizing sizing;
  sizing.num_pinhole_poses = initial_bays_from_world.size();
  sizing.num_points = num_points;
  sizing.num_sensor_from_rig_log_scales = 1;
  sizing.num_fixed_rig_pinhole = num_observations;
  sizing.num_fixed_rig_sensor_position_prior = num_priors;
  sizing.num_fixed_rig_log_scale_prior = 1;

  const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  const int gpu_index = gpu_indices.front();
  const size_t device_id = static_cast<size_t>(
      gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
  auto solver = CreateSolver(CreateSolverParameters(options), sizing, device_id);

  std::vector<StorageType> bay_data = PackPoses(initial_bays_from_world);
  std::vector<StorageType> point_data(initial_points,
                                      initial_points + 3 * num_points);
  const std::vector<StorageType> packed_sensors = PackPoses(cameras_from_bay);
  ObservationFactorData observations = BuildObservationFactorData(
      sensor_bay_indices,
      packed_sensors,
      sensor_calibrations,
      observation_sensor_indices,
      observation_point_indices,
      num_observations);
  PriorFactorData priors = BuildPriorFactorData(sensor_bay_indices,
                                                packed_sensors,
                                                prior_sensor_indices,
                                                prior_positions,
                                                prior_sqrt_information,
                                                num_priors);
  const PackedLogScale log_scale = PackedLogScale::Constant(
      static_cast<StorageType>(std::log(initial_scale)));
  const PackedLogScale packed_scale_prior_sqrt_information =
      PackedLogScale::Constant(
          static_cast<StorageType>(scale_prior_sqrt_information));
  const StorageType reprojection_loss_scale =
      options.fixed_rig_reprojection_loss_scale;

  solver.SetPinholePoseNodesFromStackedHost(
      bay_data.data(), 0, initial_bays_from_world.size());
  solver.SetPointNodesFromStackedHost(point_data.data(), 0, num_points);
  solver.SetSensorFromRigLogScaleNodesFromStackedHost(log_scale.data(), 0, 1);
  solver.SetFixedRigPinholePoseIndicesFromHost(
      observations.pose_indices.data(), num_observations);
  solver.SetFixedRigPinholePointIndicesFromHost(
      observations.point_indices.data(), num_observations);
  solver.SetFixedRigPinholeSensorFromRigDataFromStackedHost(
      observations.sensors_from_bay.data(), 0, num_observations);
  solver.SetFixedRigPinholeCalibDataFromStackedHost(
      observations.calibrations.data(), 0, num_observations);
  solver.SetFixedRigPinholePixelDataFromStackedHost(
      observation_xy, 0, num_observations);
  solver.SetFixedRigPinholeReprojectionLossScaleDataFromStackedHost(
      &reprojection_loss_scale);
  solver.SetFixedRigSensorPositionPriorPoseIndicesFromHost(
      priors.pose_indices.data(), num_priors);
  solver.SetFixedRigSensorPositionPriorSensorFromRigDataFromStackedHost(
      priors.sensors_from_bay.data(), 0, num_priors);
  solver.SetFixedRigSensorPositionPriorPositionDataFromStackedHost(
      priors.positions.data(), 0, num_priors);
  solver.SetFixedRigSensorPositionPriorSqrtInformationDataFromStackedHost(
      priors.sqrt_information.data(), 0, num_priors);
  solver.SetFixedRigLogScalePriorTargetDataFromStackedHost(log_scale.data());
  solver.SetFixedRigLogScalePriorSqrtInformationDataFromStackedHost(
      packed_scale_prior_sqrt_information.data());
  solver.SetFixedRigSchurTopology(observations.pose_indices,
                                 observations.point_indices,
                                 priors.pose_indices,
                                 /*rotation_anchor_pose_index=*/0);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve_rig_schur(
      /*print_progress=*/false,
      /*verbose_logging=*/options.collect_iteration_data);
  solver.GetPinholePoseNodesToStackedHost(
      bay_data.data(), 0, initial_bays_from_world.size());
  solver.GetPointNodesToStackedHost(point_data.data(), 0, num_points);
  PackedLogScale solved_log_scale;
  solver.GetSensorFromRigLogScaleNodesToStackedHost(
      solved_log_scale.data(), 0, 1);

  RigidBayBundleAdjustmentResult result;
  result.bays_from_world = UnpackPoses(bay_data);
  result.points = std::move(point_data);
  result.scale = std::exp(solved_log_scale[0]);
  result.summary = std::make_shared<RigidBayBundleAdjustmentSummary>(
      std::move(*CasparBundleAdjustmentSummary::Create(solve_result)));
  result.summary->num_residuals =
      static_cast<int>(2 * num_observations + 3 * num_priors + 1);
  result.summary->allocation_size = solver.get_allocation_size();
  result.summary->final_sensor_position_prior_score =
      ComputeSensorPositionPriorScore(result.bays_from_world,
                                      sensor_bay_indices,
                                      cameras_from_bay,
                                      prior_sensor_indices,
                                      prior_positions,
                                      prior_sqrt_information,
                                      num_priors,
                                      result.scale);
  const double scale_prior_residual =
      scale_prior_sqrt_information *
      (static_cast<double>(solved_log_scale[0]) - std::log(initial_scale));
  result.summary->final_scale_prior_score =
      0.5 * scale_prior_residual * scale_prior_residual;
  result.summary->final_reprojection_score =
      result.summary->final_score -
      result.summary->final_sensor_position_prior_score -
      result.summary->final_scale_prior_score;
  return result;
#endif
}

}  // namespace colmap
