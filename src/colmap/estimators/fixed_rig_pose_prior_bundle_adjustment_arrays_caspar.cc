#include "colmap/estimators/fixed_rig_pose_prior_bundle_adjustment_arrays_caspar.h"

#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/sim3.h"
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
using PackedSensorCalibration = Eigen::Matrix<StorageType, 11, 1>;
using InputCalibration = Eigen::Matrix<float, 4, 1>;
using InputSqrtInformation = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using PackedSqrtInformation = Eigen::Matrix<StorageType, 3, 3>;

struct ObservationFactorData {
  std::vector<unsigned int> pose_indices;
  std::vector<unsigned int> point_indices;
  std::vector<unsigned int> sensor_indices;
};

struct PriorFactorData {
  std::vector<unsigned int> pose_indices;
  std::vector<StorageType> positions;
  std::vector<StorageType> sqrt_information;
};

struct NormalizedProblem {
  Sim3d normalized_from_metric;
  std::vector<Rigid3d> rigs_from_world;
  std::vector<Rigid3d> sensors_from_rig;
  std::vector<StorageType> points;
  std::vector<StorageType> prior_positions;
  std::vector<StorageType> prior_sqrt_information;
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

std::vector<StorageType> PackSensorCalibrations(
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations) {
  std::vector<StorageType> packed(11 * sensors_from_rig.size());
  for (size_t sensor = 0; sensor < sensors_from_rig.size(); ++sensor) {
    Eigen::Map<PackedSensorCalibration> output(packed.data() + 11 * sensor);
    output.head<7>() = sensors_from_rig[sensor].params.cast<StorageType>();
    output.tail<4>() =
        Eigen::Map<const InputCalibration>(sensor_calibrations + 4 * sensor)
            .cast<StorageType>();
  }
  return packed;
}

void TransformPoints(std::vector<StorageType>& points,
                     const Sim3d& transformed_from_source) {
  for (size_t index = 0; index < points.size() / 3; ++index) {
    Eigen::Map<Eigen::Matrix<StorageType, 3, 1>> point(points.data() +
                                                       3 * index);
    const Eigen::Vector3d source_point = point.cast<double>();
    point = (transformed_from_source * source_point).cast<StorageType>();
  }
}

std::vector<StorageType> NormalizeSqrtInformation(
    const float* prior_sqrt_information,
    const size_t num_priors,
    const double scale) {
  std::vector<StorageType> normalized(9 * num_priors);
  for (size_t prior = 0; prior < num_priors; ++prior) {
    Eigen::Map<PackedSqrtInformation>(normalized.data() + 9 * prior) =
        Eigen::Map<const InputSqrtInformation>(prior_sqrt_information +
                                               9 * prior)
            .cast<StorageType>() /
        static_cast<StorageType>(scale);
  }
  return normalized;
}

NormalizedProblem NormalizeProblem(
    const Sim3d& normalized_from_metric,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const float* initial_points,
    const size_t num_points,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* prior_positions,
    const float* prior_sqrt_information,
    const size_t num_priors) {
  NormalizedProblem problem{
      normalized_from_metric,
      initial_rigs_from_world,
      sensors_from_rig,
      std::vector<StorageType>(initial_points, initial_points + 3 * num_points),
      std::vector<StorageType>(prior_positions,
                               prior_positions + 3 * num_priors),
      {},
  };
  for (Rigid3d& rig_from_world : problem.rigs_from_world) {
    rig_from_world =
        TransformCameraWorld(problem.normalized_from_metric, rig_from_world);
  }
  for (Rigid3d& sensor_from_rig : problem.sensors_from_rig) {
    sensor_from_rig.translation() *= problem.normalized_from_metric.scale();
  }
  TransformPoints(problem.points, problem.normalized_from_metric);
  TransformPoints(problem.prior_positions, problem.normalized_from_metric);
  problem.prior_sqrt_information =
      NormalizeSqrtInformation(prior_sqrt_information,
                               num_priors,
                               problem.normalized_from_metric.scale());
  return problem;
}

ObservationFactorData BuildObservationFactorData(
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const uint32_t* observation_image_indices,
    const uint32_t* observation_point_indices,
    const size_t num_observations) {
  ObservationFactorData data{
      std::vector<unsigned int>(num_observations),
      std::vector<unsigned int>(observation_point_indices,
                                observation_point_indices + num_observations),
      std::vector<unsigned int>(num_observations)};
  for (size_t observation = 0; observation < num_observations; ++observation) {
    const size_t image = observation_image_indices[observation];
    data.pose_indices[observation] = image_frame_indices[image];
    data.sensor_indices[observation] = image_sensor_indices[image];
  }
  return data;
}

PriorFactorData BuildPriorFactorData(
    const uint32_t* prior_frame_indices,
    std::vector<StorageType> prior_positions,
    std::vector<StorageType> prior_sqrt_information) {
  return {
      std::vector<unsigned int>(
          prior_frame_indices,
          prior_frame_indices + prior_positions.size() / 3),
      std::move(prior_positions),
      std::move(prior_sqrt_information),
  };
}

}  // namespace

FixedRigPosePriorBundleAdjustmentArraysResult
FixedRigPosePriorBundleAdjustmentArraysCaspar(
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const float* initial_points,
    const size_t num_points,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const Sim3d& normalized_from_metric,
    const float* sensor_calibrations,
    const uint32_t* observation_image_indices,
    const uint32_t* observation_point_indices,
    const float* observation_xy,
    const size_t num_observations,
    const uint32_t* prior_frame_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    const size_t num_priors,
    const CasparBundleAdjustmentOptions& options) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar fixed-rig array BA requires float precision";
  return {};
#else
  CasparSolverSizing sizing;
  sizing.num_pinhole_poses = initial_rigs_from_world.size();
  sizing.num_points = num_points;
  sizing.num_sensor_from_rig_log_scales = 1;
  sizing.num_row_fixed_rig_pinhole = num_observations;
  sizing.num_row_fixed_rig_pinhole_sensor_calibrations =
      sensors_from_rig.size();
  sizing.num_fixed_rig_position_prior = num_priors;

  const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  const int gpu_index = gpu_indices.front();
  const size_t device_id =
      static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
  auto solver =
      CreateSolver(CreateCasparSolverParameters(options), sizing, device_id);

  NormalizedProblem problem = NormalizeProblem(normalized_from_metric,
                                               initial_rigs_from_world,
                                               initial_points,
                                               num_points,
                                               sensors_from_rig,
                                               prior_positions,
                                               prior_sqrt_information,
                                               num_priors);
  std::vector<StorageType> rig_data = PackPoses(problem.rigs_from_world);
  std::vector<StorageType> point_data = std::move(problem.points);
  const std::vector<StorageType> packed_sensor_calibrations =
      PackSensorCalibrations(problem.sensors_from_rig, sensor_calibrations);
  ObservationFactorData observations =
      BuildObservationFactorData(image_frame_indices,
                                 image_sensor_indices,
                                 observation_image_indices,
                                 observation_point_indices,
                                 num_observations);
  PriorFactorData priors =
      BuildPriorFactorData(prior_frame_indices,
                           std::move(problem.prior_positions),
                           std::move(problem.prior_sqrt_information));
  const StorageType log_scale = 0;

  solver.SetPinholePoseNodesFromStackedHost(
      rig_data.data(), 0, initial_rigs_from_world.size());
  solver.SetPointNodesFromStackedHost(point_data.data(), 0, num_points);
  solver.SetSensorFromRigLogScaleNodesFromStackedHost(&log_scale, 0, 1);
  solver.SetRowFixedRigPinholePoseIndicesFromHost(
      observations.pose_indices.data(), num_observations);
  solver.SetRowFixedRigPinholePointIndicesFromHost(
      observations.point_indices.data(), num_observations);
  solver.SetRowFixedRigPinholeSensorCalibrationDataFromStackedHost(
      packed_sensor_calibrations.data(), 0, problem.sensors_from_rig.size());
  solver.SetRowFixedRigPinholeSensorCalibrationIndicesFromHost(
      observations.sensor_indices.data(), num_observations);
  solver.SetRowFixedRigPinholePixelDataFromStackedHost(
      observation_xy, 0, num_observations);
  const StorageType reprojection_loss_scale =
      options.fixed_rig_reprojection_loss_scale;
  solver.SetRowFixedRigPinholeReprojectionLossScaleDataFromStackedHost(
      &reprojection_loss_scale);
  solver.SetFixedRigPositionPriorPoseIndicesFromHost(priors.pose_indices.data(),
                                                     num_priors);
  solver.SetFixedRigPositionPriorPositionDataFromStackedHost(
      priors.positions.data(), 0, num_priors);
  solver.SetFixedRigPositionPriorSqrtInformationDataFromStackedHost(
      priors.sqrt_information.data(), 0, num_priors);
  const StorageType prior_position_loss_scale =
      options.prior_position_loss_scale;
  solver.SetFixedRigPositionPriorPositionLossScaleDataFromStackedHost(
      &prior_position_loss_scale);
  const unsigned int rotation_anchor_pose_index = *std::min_element(
      observations.pose_indices.begin(), observations.pose_indices.end());
  solver.SetRowFixedRigSchurTopology(observations.pose_indices,
                                     observations.point_indices,
                                     rotation_anchor_pose_index);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve_rig_schur(
      /*print_progress=*/false,
      /*verbose_logging=*/options.collect_iteration_data);
  solver.GetPinholePoseNodesToStackedHost(
      rig_data.data(), 0, initial_rigs_from_world.size());
  solver.GetPointNodesToStackedHost(point_data.data(), 0, num_points);
  StorageType solved_log_scale;
  solver.GetSensorFromRigLogScaleNodesToStackedHost(&solved_log_scale, 0, 1);

  const Sim3d metric_from_normalized = Inverse(problem.normalized_from_metric);
  std::vector<Rigid3d> solved_rigs_from_world = UnpackPoses(rig_data);
  for (Rigid3d& rig_from_world : solved_rigs_from_world) {
    rig_from_world =
        TransformCameraWorld(metric_from_normalized, rig_from_world);
  }
  TransformPoints(point_data, metric_from_normalized);

  FixedRigPosePriorBundleAdjustmentArraysResult result;
  result.rigs_from_world = std::move(solved_rigs_from_world);
  result.points = std::move(point_data);
  result.sensor_from_rig_scale = std::exp(solved_log_scale);
  result.summary = CasparBundleAdjustmentSummary::Create(solve_result);
  result.summary->num_residuals =
      static_cast<int>(2 * num_observations + 3 * num_priors);
  result.summary->allocation_size = solver.get_allocation_size();
  return result;
#endif
}

}  // namespace colmap
