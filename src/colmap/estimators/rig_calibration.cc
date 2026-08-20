// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/rig_calibration.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <ceres/crs_matrix.h>

namespace colmap {
namespace {

constexpr double kObservabilityRankTolerance = 1e-10;
constexpr double kMaxReprojectionError = 4.0;
constexpr double kMinTriangulationAngleDeg = 1.5;

class RigCenterDistanceCostFunctor
    : public AutoDiffCostFunctor<RigCenterDistanceCostFunctor, 1, 7, 7> {
 public:
  RigCenterDistanceCostFunctor(const double distance, const double stddev)
      : distance_(distance), inv_stddev_(1.0 / stddev) {}

  template <typename T>
  bool operator()(const T* const rig1_from_group,
                  const T* const rig2_from_group,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> center1 =
        EigenQuaternionMap<T>(rig1_from_group).inverse() *
        -EigenVector3Map<T>(rig1_from_group + 4);
    const Eigen::Matrix<T, 3, 1> center2 =
        EigenQuaternionMap<T>(rig2_from_group).inverse() *
        -EigenVector3Map<T>(rig2_from_group + 4);
    residuals[0] = ((center2 - center1).norm() - T(distance_)) * T(inv_stddev_);
    return true;
  }

 private:
  double distance_;
  double inv_stddev_;
};

std::unique_ptr<ceres::LossFunction> CreateLossFunction(
    const CeresBundleAdjustmentOptions::LossFunctionType type,
    const double scale) {
  switch (type) {
    case CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL:
      return std::make_unique<ceres::TrivialLoss>();
    case CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1:
      return std::make_unique<ceres::SoftLOneLoss>(scale);
    case CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY:
      return std::make_unique<ceres::CauchyLoss>(scale);
    case CeresBundleAdjustmentOptions::LossFunctionType::HUBER:
      return std::make_unique<ceres::HuberLoss>(scale);
  }
  LOG(FATAL_THROW) << "Invalid loss function type";
  return nullptr;
}

std::vector<int> ConstantCameraParams(const RigCalibrationOptions& options,
                                      const Camera& camera,
                                      const bool refine_distortion) {
  std::vector<int> constant_params;
  const auto append = [&constant_params](const span<const size_t> indices) {
    for (const size_t index : indices) {
      constant_params.push_back(static_cast<int>(index));
    }
  };
  append(camera.MetaDataParamsIdxs());
  if (!options.refine_focal_length) {
    append(camera.FocalLengthIdxs());
  }
  if (!options.refine_principal_point) {
    append(camera.PrincipalPointIdxs());
  }
  if (!refine_distortion) {
    append(camera.ExtraParamsIdxs());
  } else if (camera.model_id == FullOpenCVCameraModel::model_id) {
    constant_params.insert(
        constant_params.end(),
        {static_cast<int>(FullOpenCVCameraModel::extra_params_idxs[5]),
         static_cast<int>(FullOpenCVCameraModel::extra_params_idxs[6]),
         static_cast<int>(FullOpenCVCameraModel::extra_params_idxs[7])});
  }
  std::sort(constant_params.begin(), constant_params.end());
  constant_params.erase(
      std::unique(constant_params.begin(), constant_params.end()),
      constant_params.end());
  return constant_params;
}

Eigen::MatrixXd SymmetricPseudoInverse(const Eigen::MatrixXd& matrix,
                                       const double relative_tolerance) {
  if (matrix.rows() == 0) {
    return matrix;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(matrix);
  THROW_CHECK_EQ(eig.info(), Eigen::Success);
  const double max_eigenvalue = std::max(0.0, eig.eigenvalues().maxCoeff());
  const double threshold = relative_tolerance * max_eigenvalue;
  Eigen::VectorXd inv_eigenvalues = Eigen::VectorXd::Zero(matrix.rows());
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    if (eig.eigenvalues()[i] > threshold) {
      inv_eigenvalues[i] = 1.0 / eig.eigenvalues()[i];
    }
  }
  return eig.eigenvectors() * inv_eigenvalues.asDiagonal() *
         eig.eigenvectors().transpose();
}

bool ShouldRetryWithCpu(const ceres::Solver::Summary& summary) {
  if (summary.termination_type != ceres::FAILURE) {
    return false;
  }
  return summary.message.find("CUDA initialization failed") !=
             std::string::npos ||
         summary.message.find("non-numeric") != std::string::npos ||
         summary.message.find("Unable to create Jacobian") != std::string::npos;
}

}  // namespace

RigCalibrationOptions::RigCalibrationOptions() {
  ceres.loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
  ceres.loss_function_scale = 1.0;
  ceres.auto_select_solver_type = false;
  ceres.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

bool RigCalibrationOptions::Check() const {
  CHECK_OPTION_GT(distance_loss_function_scale, 0);
  return ceres.Check();
}

bool RigCalibrationObservability::IsFullRank() const {
  return rank == parameter_names.size();
}

struct CeresRigCalibrator::Impl {
  struct GroupProblemData {
    std::vector<ceres::ResidualBlockId> residual_blocks;
    std::vector<double*> local_pose_blocks;
    std::vector<double*> point_blocks;
  };

  Impl(const RigCalibrationOptions& options,
       const rig_t rig_id,
       std::vector<RigCalibrationGroup> groups,
       Reconstruction& reconstruction)
      : options(options),
        rig_id(rig_id),
        groups(std::move(groups)),
        reconstruction(reconstruction),
        reprojection_loss(options.ceres.CreateLossFunction()),
        distance_loss(
            CreateLossFunction(options.distance_loss_function_type,
                               options.distance_loss_function_scale)) {
    THROW_CHECK(options.Check());
    THROW_CHECK(reconstruction.ExistsRig(rig_id));
    THROW_CHECK_GT(this->groups.size(), 0);
    PrepareGroups();
    BuildProblem(CalibrationStage::LOCAL);
  }

  bool HasRequestedGlobalCalibration() const {
    return options.refine_focal_length || options.refine_principal_point ||
           options.refine_distortion || options.refine_sensor_from_rig;
  }

  void PrepareGroups() {
    Rig& rig = reconstruction.Rig(rig_id);
    for (const sensor_t sensor_id : rig.SensorIds()) {
      THROW_CHECK_EQ(sensor_id.type, SensorType::CAMERA);
      const camera_t camera_id = sensor_id.id;
      THROW_CHECK(reconstruction.ExistsCamera(camera_id));
      rig_camera_ids.insert(camera_id);
      if (!rig.IsRefSensor(sensor_id)) {
        THROW_CHECK(rig.HasSensorFromRig(sensor_id));
      }
    }
    for (RigCalibrationGroup& group : groups) {
      THROW_CHECK_GT(group.frame0_to_frame2_distance.distance, 0);
      THROW_CHECK_GT(group.frame0_to_frame2_distance.stddev, 0);
      THROW_CHECK_GT(group.tracks.size(), 0);

      std::array<bool, 3> frame_has_observation = {false, false, false};
      for (Rigid3d& rig_from_group : group.rigs_from_group) {
        THROW_CHECK(rig_from_group.params.allFinite());
        THROW_CHECK_GT(rig_from_group.rotation().norm(), 0);
        rig_from_group.rotation().normalize();
      }
      THROW_CHECK_GT((group.rigs_from_group[2].TgtOriginInSrc() -
                      group.rigs_from_group[0].TgtOriginInSrc())
                         .norm(),
                     0);
      for (const RigCalibrationTrack& track : group.tracks) {
        THROW_CHECK(track.xyz.allFinite());
        THROW_CHECK_GE(track.observations.size(), 2);
        std::set<std::pair<size_t, camera_t>> observed_images;
        for (const RigCalibrationObservation& observation :
             track.observations) {
          THROW_CHECK_LT(observation.frame_idx, 3);
          THROW_CHECK(observation.xy.allFinite());
          THROW_CHECK(reconstruction.ExistsCamera(observation.camera_id));
          const Camera& camera = reconstruction.Camera(observation.camera_id);
          THROW_CHECK(rig.HasSensor(camera.SensorId()));
          if (!rig.IsRefSensor(camera.SensorId())) {
            THROW_CHECK(rig.HasSensorFromRig(camera.SensorId()));
          }
          THROW_CHECK(observed_images
                          .emplace(observation.frame_idx, observation.camera_id)
                          .second);
          frame_has_observation[observation.frame_idx] = true;
        }
      }
      THROW_CHECK(std::all_of(frame_has_observation.begin(),
                              frame_has_observation.end(),
                              [](const bool value) { return value; }));

      const Rigid3d frame0_from_old_group = group.rigs_from_group[0];
      for (Rigid3d& rig_from_old_group : group.rigs_from_group) {
        rig_from_old_group =
            rig_from_old_group * Inverse(frame0_from_old_group);
      }
      group.rigs_from_group[0] = Rigid3d();
      for (RigCalibrationTrack& track : group.tracks) {
        track.xyz = frame0_from_old_group * track.xyz;
      }
    }
  }

  void ParameterizeCamera(Camera& camera,
                          const bool refine_global_calibration) {
    if (!refine_global_calibration) {
      problem->SetParameterBlockConstant(camera.params.data());
      return;
    }
    const std::vector<int> constant_params =
        ConstantCameraParams(options, camera, options.refine_distortion);
    if (constant_params.size() == camera.params.size()) {
      problem->SetParameterBlockConstant(camera.params.data());
    } else if (!constant_params.empty()) {
      SetManifold(problem.get(),
                  camera.params.data(),
                  CreateSubsetManifold(camera.params.size(), constant_params));
    }
  }

  enum class CalibrationStage { LOCAL, ROBUST_JOINT, FINAL_JOINT };

  void BuildProblem(const CalibrationStage stage) {
    const bool refine_global_calibration = stage != CalibrationStage::LOCAL;
    ceres::LossFunction* reprojection_loss_function =
        stage == CalibrationStage::FINAL_JOINT ? nullptr
                                               : reprojection_loss.get();
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem = std::make_shared<ceres::Problem>(problem_options);
    ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    group_problem_data.clear();
    reprojection_residual_blocks.clear();
    global_parameter_blocks.clear();
    global_parameter_names.clear();
    group_problem_data.resize(groups.size());

    Rig& rig = reconstruction.Rig(rig_id);
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      RigCalibrationGroup& group = groups[group_idx];
      GroupProblemData& group_data = group_problem_data[group_idx];
      for (RigCalibrationTrack& track : group.tracks) {
        group_data.point_blocks.push_back(track.xyz.data());
        for (const RigCalibrationObservation& observation :
             track.observations) {
          Camera& camera = reconstruction.Camera(observation.camera_id);
          Rigid3d& rig_from_group =
              group.rigs_from_group[observation.frame_idx];
          ceres::ResidualBlockId residual_block;
          if (rig.IsRefSensor(camera.SensorId())) {
            residual_block = problem->AddResidualBlock(
                CreateCameraCostFunction<ReprojErrorCostFunctor>(
                    camera.model_id, observation.xy),
                reprojection_loss_function,
                track.xyz.data(),
                rig_from_group.params.data(),
                camera.params.data());
          } else {
            Rigid3d& sensor_from_rig = rig.SensorFromRig(camera.SensorId());
            residual_block = problem->AddResidualBlock(
                CreateCameraCostFunction<RigReprojErrorCostFunctor>(
                    camera.model_id, observation.xy),
                reprojection_loss_function,
                track.xyz.data(),
                sensor_from_rig.params.data(),
                rig_from_group.params.data(),
                camera.params.data());
          }
          reprojection_residual_blocks.push_back(residual_block);
          group_data.residual_blocks.push_back(residual_block);
        }
      }

      const RigCalibrationDistancePrior& prior =
          group.frame0_to_frame2_distance;
      const ceres::ResidualBlockId distance_residual =
          problem->AddResidualBlock(RigCenterDistanceCostFunctor::Create(
                                        prior.distance, prior.stddev),
                                    distance_loss.get(),
                                    group.rigs_from_group[0].params.data(),
                                    group.rigs_from_group[2].params.data());
      group_data.residual_blocks.push_back(distance_residual);

      for (size_t frame_idx = 0; frame_idx < 3; ++frame_idx) {
        Rigid3d& rig_from_group = group.rigs_from_group[frame_idx];
        SetManifold(problem.get(),
                    rig_from_group.params.data(),
                    CreateProductManifold(CreateEigenQuaternionManifold(),
                                          CreateEuclideanManifold<3>()));
        if (frame_idx == 0) {
          problem->SetParameterBlockConstant(rig_from_group.params.data());
        } else {
          group_data.local_pose_blocks.push_back(rig_from_group.params.data());
        }
      }
    }

    for (const auto& [sensor_id, maybe_sensor_from_rig] : rig.NonRefSensors()) {
      THROW_CHECK(maybe_sensor_from_rig.has_value());
      Rigid3d& sensor_from_rig = rig.SensorFromRig(sensor_id);
      if (!problem->HasParameterBlock(sensor_from_rig.params.data())) {
        problem->AddParameterBlock(sensor_from_rig.params.data(), 7);
      }
      SetManifold(problem.get(),
                  sensor_from_rig.params.data(),
                  CreateProductManifold(CreateEigenQuaternionManifold(),
                                        CreateEuclideanManifold<3>()));
      if (refine_global_calibration && options.refine_sensor_from_rig) {
        global_parameter_blocks.push_back(sensor_from_rig.params.data());
        const std::string prefix =
            StringPrintf("sensor_from_rig[camera:%u].", sensor_id.id);
        global_parameter_names.insert(global_parameter_names.end(),
                                      {prefix + "rotation_x",
                                       prefix + "rotation_y",
                                       prefix + "rotation_z",
                                       prefix + "translation_x",
                                       prefix + "translation_y",
                                       prefix + "translation_z"});
      } else {
        problem->SetParameterBlockConstant(sensor_from_rig.params.data());
      }
    }

    for (const camera_t camera_id : rig_camera_ids) {
      Camera& camera = reconstruction.Camera(camera_id);
      if (!problem->HasParameterBlock(camera.params.data())) {
        problem->AddParameterBlock(camera.params.data(), camera.params.size());
      }
      ParameterizeCamera(camera, refine_global_calibration);
      if (!problem->IsParameterBlockConstant(camera.params.data())) {
        global_parameter_blocks.push_back(camera.params.data());
        const std::vector<int> constant_params =
            ConstantCameraParams(options, camera, options.refine_distortion);
        const std::unordered_set<int> constant_params_set(
            constant_params.begin(), constant_params.end());
        const std::vector<std::string> param_names =
            CSVToVector<std::string>(camera.ParamsInfo());
        for (size_t param_idx = 0; param_idx < camera.params.size();
             ++param_idx) {
          if (constant_params_set.count(static_cast<int>(param_idx)) != 0) {
            continue;
          }
          const std::string param_name =
              param_idx < param_names.size()
                  ? param_names[param_idx]
                  : StringPrintf("param_%zu", param_idx);
          global_parameter_names.push_back(
              StringPrintf("camera[%u].%s", camera_id, param_name.c_str()));
        }
      }
    }

    for (const GroupProblemData& group_data : group_problem_data) {
      for (double* point_block : group_data.point_blocks) {
        ordering->AddElementToGroup(point_block, 0);
      }
      for (double* pose_block : group_data.local_pose_blocks) {
        ordering->AddElementToGroup(pose_block, 1);
      }
    }
    for (double* global_block : global_parameter_blocks) {
      ordering->AddElementToGroup(global_block, 2);
    }
  }

  ceres::Solver::Summary SolveProblem() {
    ceres::Solver::Options solver_options =
        options.ceres.CreateSolverOptions(3 * groups.size(), *problem);
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.linear_solver_ordering = ordering;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, problem.get(), &summary);
    if (options.ceres.use_gpu && ShouldRetryWithCpu(summary)) {
      LOG(WARNING) << "GPU rig calibration failed (" << summary.message
                   << "), retrying with CPU.";
      CeresBundleAdjustmentOptions cpu_options = options.ceres;
      cpu_options.use_gpu = false;
      solver_options =
          cpu_options.CreateSolverOptions(3 * groups.size(), *problem);
      solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
      solver_options.linear_solver_ordering = ordering;
      ceres::Solve(solver_options, problem.get(), &summary);
    }
    return summary;
  }

  bool IsObservationOutlier(
      const RigCalibrationGroup& group,
      const RigCalibrationTrack& track,
      const RigCalibrationObservation& observation) const {
    const Rig& rig = reconstruction.Rig(rig_id);
    const Camera& camera = reconstruction.Camera(observation.camera_id);
    Eigen::Vector3d point_in_cam =
        group.rigs_from_group[observation.frame_idx] * track.xyz;
    if (!rig.IsRefSensor(camera.SensorId())) {
      point_in_cam = rig.SensorFromRig(camera.SensorId()) * point_in_cam;
    }
    const std::optional<Eigen::Vector2d> predicted_xy =
        camera.ImgFromCam(point_in_cam);
    if (!predicted_xy) {
      return true;
    }
    return (*predicted_xy - observation.xy).norm() > kMaxReprojectionError;
  }

  double MaxTriangulationAngleDeg(const RigCalibrationGroup& group,
                                  const RigCalibrationTrack& track) const {
    const Rig& rig = reconstruction.Rig(rig_id);
    std::vector<Eigen::Vector3d> rays;
    rays.reserve(track.observations.size());
    for (const RigCalibrationObservation& observation : track.observations) {
      const Camera& camera = reconstruction.Camera(observation.camera_id);
      Rigid3d cam_from_group = group.rigs_from_group[observation.frame_idx];
      if (!rig.IsRefSensor(camera.SensorId())) {
        cam_from_group = rig.SensorFromRig(camera.SensorId()) * cam_from_group;
      }
      rays.push_back(
          (track.xyz - cam_from_group.TgtOriginInSrc()).normalized());
    }
    double max_angle_deg = 0.0;
    for (size_t i = 0; i < rays.size(); ++i) {
      for (size_t j = i + 1; j < rays.size(); ++j) {
        max_angle_deg = std::max(
            max_angle_deg,
            RadToDeg(std::acos(std::clamp(rays[i].dot(rays[j]), -1.0, 1.0))));
      }
    }
    return max_angle_deg;
  }

  size_t FilterOutlierObservations(size_t* num_filtered_groups) {
    problem.reset();
    ordering.reset();
    group_problem_data.clear();
    reprojection_residual_blocks.clear();

    size_t num_filtered_observations = 0;
    for (RigCalibrationGroup& group : groups) {
      for (RigCalibrationTrack& track : group.tracks) {
        const auto new_end = std::remove_if(
            track.observations.begin(),
            track.observations.end(),
            [this, &group, &track](
                const RigCalibrationObservation& observation) {
              return IsObservationOutlier(group, track, observation);
            });
        num_filtered_observations +=
            std::distance(new_end, track.observations.end());
        track.observations.erase(new_end, track.observations.end());
      }

      const auto new_track_end = std::remove_if(
          group.tracks.begin(),
          group.tracks.end(),
          [this, &group, &num_filtered_observations](
              const RigCalibrationTrack& track) {
            if (track.observations.size() >= 2 &&
                MaxTriangulationAngleDeg(group, track) >=
                    kMinTriangulationAngleDeg) {
              return false;
            }
            num_filtered_observations += track.observations.size();
            return true;
          });
      group.tracks.erase(new_track_end, group.tracks.end());
    }

    const auto new_group_end = std::remove_if(
        groups.begin(),
        groups.end(),
        [&num_filtered_observations](const RigCalibrationGroup& group) {
          std::array<bool, 3> frame_has_observation = {false, false, false};
          for (const RigCalibrationTrack& track : group.tracks) {
            for (const RigCalibrationObservation& observation :
                 track.observations) {
              frame_has_observation[observation.frame_idx] = true;
            }
          }
          if (!group.tracks.empty() &&
              std::all_of(frame_has_observation.begin(),
                          frame_has_observation.end(),
                          [](const bool value) { return value; })) {
            return false;
          }
          for (const RigCalibrationTrack& track : group.tracks) {
            num_filtered_observations += track.observations.size();
          }
          return true;
        });
    *num_filtered_groups = std::distance(new_group_end, groups.end());
    groups.erase(new_group_end, groups.end());
    return num_filtered_observations;
  }

  RigCalibrationObservability RejectedObservability() const {
    RigCalibrationObservability observability;
    observability.parameter_names = global_parameter_names;
    const Eigen::Index num_params = global_parameter_names.size();
    observability.marginal_information =
        Eigen::MatrixXd::Zero(num_params, num_params);
    observability.marginal_covariance =
        Eigen::MatrixXd::Zero(num_params, num_params);
    observability.standard_deviations = Eigen::VectorXd::Constant(
        num_params, std::numeric_limits<double>::infinity());
    observability.normalized_information_eigenvalues =
        Eigen::VectorXd::Zero(num_params);
    return observability;
  }

  void ComputeResidualStatistics(RigCalibrationSummary* summary) const {
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.apply_loss_function = false;
    eval_options.residual_blocks = reprojection_residual_blocks;
    std::vector<double> residuals;
    THROW_CHECK(
        problem->Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr));
    THROW_CHECK_EQ(residuals.size(), 2 * reprojection_residual_blocks.size());

    summary->reprojection_errors.resize(reprojection_residual_blocks.size());
    for (size_t i = 0; i < summary->reprojection_errors.size(); ++i) {
      summary->reprojection_errors[i] =
          std::hypot(residuals[2 * i], residuals[2 * i + 1]);
    }
    const double squared_reprojection_error =
        std::accumulate(summary->reprojection_errors.begin(),
                        summary->reprojection_errors.end(),
                        0.0,
                        [](const double sum, const double error) {
                          return sum + error * error;
                        });
    summary->reprojection_rmse = std::sqrt(squared_reprojection_error /
                                           summary->reprojection_errors.size());

    summary->distance_prior_errors.reserve(groups.size());
    double squared_distance_error = 0.0;
    for (const RigCalibrationGroup& group : groups) {
      const double error = (group.rigs_from_group[2].TgtOriginInSrc() -
                            group.rigs_from_group[0].TgtOriginInSrc())
                               .norm() -
                           group.frame0_to_frame2_distance.distance;
      summary->distance_prior_errors.push_back(error);
      squared_distance_error += error * error;
    }
    summary->distance_prior_rmse =
        std::sqrt(squared_distance_error / groups.size());
  }

  RigCalibrationObservability ComputeObservability() const {
    RigCalibrationObservability observability;
    observability.parameter_names = global_parameter_names;
    if (global_parameter_blocks.empty()) {
      observability.marginal_information = Eigen::MatrixXd(0, 0);
      observability.marginal_covariance = Eigen::MatrixXd(0, 0);
      observability.standard_deviations = Eigen::VectorXd(0);
      observability.normalized_information_eigenvalues = Eigen::VectorXd(0);
      observability.normalized_condition_number = 1.0;
      return observability;
    }

    int num_global_params = 0;
    for (const double* global_block : global_parameter_blocks) {
      num_global_params += ParameterBlockTangentSize(*problem, global_block);
    }
    THROW_CHECK_EQ(num_global_params, global_parameter_names.size());
    Eigen::MatrixXd marginal_information =
        Eigen::MatrixXd::Zero(num_global_params, num_global_params);

    for (const GroupProblemData& group_data : group_problem_data) {
      ceres::Problem::EvaluateOptions eval_options;
      eval_options.apply_loss_function = true;
      eval_options.residual_blocks = group_data.residual_blocks;
      eval_options.parameter_blocks = global_parameter_blocks;
      eval_options.parameter_blocks.insert(eval_options.parameter_blocks.end(),
                                           group_data.local_pose_blocks.begin(),
                                           group_data.local_pose_blocks.end());
      eval_options.parameter_blocks.insert(eval_options.parameter_blocks.end(),
                                           group_data.point_blocks.begin(),
                                           group_data.point_blocks.end());

      ceres::CRSMatrix jacobian_crs;
      THROW_CHECK(problem->Evaluate(
          eval_options, nullptr, nullptr, nullptr, &jacobian_crs));
      const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>>
          jacobian_map(jacobian_crs.num_rows,
                       jacobian_crs.num_cols,
                       jacobian_crs.values.size(),
                       jacobian_crs.rows.data(),
                       jacobian_crs.cols.data(),
                       jacobian_crs.values.data());
      const Eigen::SparseMatrix<double> jacobian = jacobian_map;

      int num_local_pose_params = 0;
      for (const double* pose_block : group_data.local_pose_blocks) {
        num_local_pose_params +=
            ParameterBlockTangentSize(*problem, pose_block);
      }
      const int num_a_params = num_global_params + num_local_pose_params;
      const int num_point_params =
          static_cast<int>(3 * group_data.point_blocks.size());
      THROW_CHECK_EQ(jacobian.cols(), num_a_params + num_point_params);

      const Eigen::SparseMatrix<double> jacobian_a =
          jacobian.leftCols(num_a_params);
      const Eigen::SparseMatrix<double> jacobian_points =
          jacobian.rightCols(num_point_params);
      Eigen::MatrixXd schur =
          Eigen::MatrixXd(jacobian_a.transpose() * jacobian_a);
      const Eigen::MatrixXd hessian_a_points =
          Eigen::MatrixXd(jacobian_a.transpose() * jacobian_points);
      const Eigen::SparseMatrix<double> hessian_points =
          jacobian_points.transpose() * jacobian_points;
      for (size_t point_idx = 0; point_idx < group_data.point_blocks.size();
           ++point_idx) {
        const Eigen::Index offset = 3 * point_idx;
        const Eigen::Matrix3d point_hessian =
            Eigen::MatrixXd(hessian_points.block(offset, offset, 3, 3));
        const Eigen::Matrix3d point_hessian_inv =
            SymmetricPseudoInverse(point_hessian, kObservabilityRankTolerance);
        const Eigen::MatrixXd cross = hessian_a_points.middleCols(offset, 3);
        schur.noalias() -= cross * point_hessian_inv * cross.transpose();
      }
      schur = 0.5 * (schur + schur.transpose());

      const Eigen::MatrixXd global_information =
          schur.topLeftCorner(num_global_params, num_global_params);
      if (num_local_pose_params == 0) {
        marginal_information += global_information;
      } else {
        const Eigen::MatrixXd global_local =
            schur.topRightCorner(num_global_params, num_local_pose_params);
        const Eigen::MatrixXd local_information = schur.bottomRightCorner(
            num_local_pose_params, num_local_pose_params);
        marginal_information.noalias() +=
            global_information -
            global_local *
                SymmetricPseudoInverse(local_information,
                                       kObservabilityRankTolerance) *
                global_local.transpose();
      }
    }
    marginal_information =
        0.5 * (marginal_information + marginal_information.transpose());
    observability.marginal_information = marginal_information;

    Eigen::VectorXd scales(num_global_params);
    for (int i = 0; i < num_global_params; ++i) {
      scales[i] = marginal_information(i, i) > 0
                      ? 1.0 / std::sqrt(marginal_information(i, i))
                      : 0.0;
    }
    const Eigen::MatrixXd normalized_information =
        scales.asDiagonal() * marginal_information * scales.asDiagonal();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(
        normalized_information);
    THROW_CHECK_EQ(eig.info(), Eigen::Success);
    observability.normalized_information_eigenvalues = eig.eigenvalues();
    const double max_eigenvalue = std::max(0.0, eig.eigenvalues().maxCoeff());
    const double threshold = kObservabilityRankTolerance * max_eigenvalue;
    double min_nonzero_eigenvalue = std::numeric_limits<double>::infinity();
    Eigen::VectorXd inv_eigenvalues = Eigen::VectorXd::Zero(num_global_params);
    for (int i = 0; i < num_global_params; ++i) {
      if (eig.eigenvalues()[i] > threshold) {
        ++observability.rank;
        min_nonzero_eigenvalue =
            std::min(min_nonzero_eigenvalue, eig.eigenvalues()[i]);
        inv_eigenvalues[i] = 1.0 / eig.eigenvalues()[i];
      }
    }
    if (observability.rank > 0) {
      observability.normalized_condition_number =
          max_eigenvalue / min_nonzero_eigenvalue;
    }
    observability.marginal_covariance =
        scales.asDiagonal() * eig.eigenvectors() *
        inv_eigenvalues.asDiagonal() * eig.eigenvectors().transpose() *
        scales.asDiagonal();
    observability.standard_deviations =
        observability.marginal_covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
    return observability;
  }

  std::shared_ptr<RigCalibrationSummary> Solve() {
    std::vector<ceres::Solver::Summary> stage_summaries;
    ceres::Solver::Summary summary = SolveProblem();
    stage_summaries.push_back(summary);

    size_t num_filtered_groups = 0;
    size_t num_filtered_observations = 0;
    bool rejected_all_groups = false;
    if (summary.IsSolutionUsable() && HasRequestedGlobalCalibration()) {
      BuildProblem(CalibrationStage::ROBUST_JOINT);
      summary = SolveProblem();
      stage_summaries.push_back(summary);
      if (summary.IsSolutionUsable()) {
        num_filtered_observations =
            FilterOutlierObservations(&num_filtered_groups);
        if (groups.empty()) {
          rejected_all_groups = true;
          summary.termination_type = ceres::FAILURE;
          summary.message = "All calibration groups were rejected";
        } else {
          BuildProblem(CalibrationStage::FINAL_JOINT);
          summary = SolveProblem();
          stage_summaries.push_back(summary);
        }
      }
    }

    auto result = std::make_shared<RigCalibrationSummary>();
    const auto base_summary = CeresBundleAdjustmentSummary::Create(summary);
    result->termination_type = base_summary->termination_type;
    result->num_residuals = base_summary->num_residuals;
    result->ceres_summary = summary;
    result->stage_summaries = std::move(stage_summaries);
    result->num_groups = groups.size();
    result->num_filtered_groups = num_filtered_groups;
    result->num_filtered_observations = num_filtered_observations;
    for (const RigCalibrationGroup& group : groups) {
      result->num_tracks += group.tracks.size();
      for (const RigCalibrationTrack& track : group.tracks) {
        result->num_observations += track.observations.size();
      }
    }
    if (rejected_all_groups || !summary.IsSolutionUsable()) {
      result->reprojection_rmse = std::numeric_limits<double>::infinity();
      result->distance_prior_rmse = std::numeric_limits<double>::infinity();
      result->observability = RejectedObservability();
    } else {
      ComputeResidualStatistics(result.get());
      result->observability = ComputeObservability();
    }

    if (options.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(summary, "Rig calibration report");
    }
    return result;
  }

  const RigCalibrationOptions options;
  const rig_t rig_id;
  std::vector<RigCalibrationGroup> groups;
  Reconstruction& reconstruction;
  std::unique_ptr<ceres::LossFunction> reprojection_loss;
  std::unique_ptr<ceres::LossFunction> distance_loss;
  std::set<camera_t> rig_camera_ids;

  std::shared_ptr<ceres::Problem> problem;
  std::shared_ptr<ceres::ParameterBlockOrdering> ordering;
  std::vector<GroupProblemData> group_problem_data;
  std::vector<ceres::ResidualBlockId> reprojection_residual_blocks;
  std::vector<double*> global_parameter_blocks;
  std::vector<std::string> global_parameter_names;
};

CeresRigCalibrator::CeresRigCalibrator(const RigCalibrationOptions& options,
                                       const rig_t rig_id,
                                       std::vector<RigCalibrationGroup> groups,
                                       Reconstruction& reconstruction)
    : impl_(std::make_unique<Impl>(
          options, rig_id, std::move(groups), reconstruction)) {}

CeresRigCalibrator::~CeresRigCalibrator() = default;

std::shared_ptr<RigCalibrationSummary> CeresRigCalibrator::Solve() {
  return impl_->Solve();
}

std::shared_ptr<ceres::Problem>& CeresRigCalibrator::Problem() {
  return impl_->problem;
}

const RigCalibrationOptions& CeresRigCalibrator::Options() const {
  return impl_->options;
}

std::unique_ptr<CeresRigCalibrator> CreateCeresRigCalibrator(
    const RigCalibrationOptions& options,
    const rig_t rig_id,
    std::vector<RigCalibrationGroup> groups,
    Reconstruction& reconstruction) {
  return std::make_unique<CeresRigCalibrator>(
      options, rig_id, std::move(groups), reconstruction);
}

}  // namespace colmap
