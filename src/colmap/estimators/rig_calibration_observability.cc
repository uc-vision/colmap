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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/rig_calibration_observability.h"

#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <numeric>
#include <utility>

#include <Eigen/Eigenvalues>

namespace colmap {
namespace internal {
namespace {

Eigen::MatrixXd SymmetricPseudoInverse(const Eigen::MatrixXd& matrix,
                                       const double relative_tolerance) {
  if (matrix.rows() == 0) {
    return matrix;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(matrix);
  THROW_CHECK_EQ(eig.info(), Eigen::Success);
  const double max_eigenvalue = std::max(0.0, eig.eigenvalues().maxCoeff());
  const double threshold = relative_tolerance * max_eigenvalue;
  Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(matrix.rows());
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    if (eig.eigenvalues()[i] > threshold) {
      inverse_eigenvalues[i] = 1.0 / eig.eigenvalues()[i];
    }
  }
  return eig.eigenvectors() * inverse_eigenvalues.asDiagonal() *
         eig.eigenvectors().transpose();
}

Eigen::Matrix3d SymmetricPseudoInverse(const Eigen::Matrix3d& matrix,
                                       const double relative_tolerance) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(matrix);
  THROW_CHECK_EQ(eig.info(), Eigen::Success);
  const double max_eigenvalue = std::max(0.0, eig.eigenvalues().maxCoeff());
  const double threshold = relative_tolerance * max_eigenvalue;
  Eigen::Vector3d inverse_eigenvalues = Eigen::Vector3d::Zero();
  for (Eigen::Index i = 0; i < inverse_eigenvalues.size(); ++i) {
    if (eig.eigenvalues()[i] > threshold) {
      inverse_eigenvalues[i] = 1.0 / eig.eigenvalues()[i];
    }
  }
  return eig.eigenvectors() * inverse_eigenvalues.asDiagonal() *
         eig.eigenvectors().transpose();
}

void AddRowOuterProduct(
    const std::vector<std::pair<int, double>>& jacobian_entries,
    Eigen::MatrixXd* lower_information) {
  for (size_t entry_idx = 0; entry_idx < jacobian_entries.size(); ++entry_idx) {
    const auto [column1, value1] = jacobian_entries[entry_idx];
    for (size_t other_idx = 0; other_idx <= entry_idx; ++other_idx) {
      const auto [column2, value2] = jacobian_entries[other_idx];
      (*lower_information)(std::max(column1, column2),
                           std::min(column1, column2)) += value1 * value2;
    }
  }
}

struct EvaluatedGroupJacobian {
  ceres::CRSMatrix jacobian;
  std::vector<int> track_row_counts;
};

EvaluatedGroupJacobian EvaluateGroupJacobian(
    const ceres::Problem& source_problem,
    const std::vector<double*>& global_parameter_blocks,
    const RigCalibrationObservabilityGroupData& group_data,
    ceres::Context* context) {
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.context = context;
  ceres::Problem group_problem(problem_options);

  std::vector<double*> variable_blocks;
  variable_blocks.reserve(global_parameter_blocks.size() +
                          group_data.local_pose_blocks.size() +
                          group_data.tracks.size());
  const auto add_variable_block = [&](double* parameter_block) {
    group_problem.AddParameterBlock(
        parameter_block,
        source_problem.ParameterBlockSize(parameter_block),
        const_cast<ceres::Manifold*>(
            source_problem.GetManifold(parameter_block)));
    variable_blocks.push_back(parameter_block);
  };
  for (double* parameter_block : global_parameter_blocks) {
    add_variable_block(parameter_block);
  }
  for (double* parameter_block : group_data.local_pose_blocks) {
    add_variable_block(parameter_block);
  }
  for (const RigCalibrationObservabilityTrackData& track : group_data.tracks) {
    add_variable_block(track.point_block);
  }

  std::vector<ceres::ResidualBlockId> residual_blocks;
  residual_blocks.reserve(group_data.track_residual_blocks.size() +
                          group_data.non_track_residual_blocks.size());
  std::vector<double*> residual_parameter_blocks;
  const auto add_residual = [&](const ceres::ResidualBlockId source_residual) {
    source_problem.GetParameterBlocksForResidualBlock(
        source_residual, &residual_parameter_blocks);
    for (double* parameter_block : residual_parameter_blocks) {
      if (!group_problem.HasParameterBlock(parameter_block)) {
        group_problem.AddParameterBlock(
            parameter_block,
            source_problem.ParameterBlockSize(parameter_block));
        group_problem.SetParameterBlockConstant(parameter_block);
      }
    }
    const ceres::CostFunction* cost_function =
        source_problem.GetCostFunctionForResidualBlock(source_residual);
    residual_blocks.push_back(group_problem.AddResidualBlock(
        const_cast<ceres::CostFunction*>(cost_function),
        const_cast<ceres::LossFunction*>(
            source_problem.GetLossFunctionForResidualBlock(source_residual)),
        residual_parameter_blocks));
    return cost_function->num_residuals();
  };

  std::vector<int> track_row_counts;
  track_row_counts.reserve(group_data.tracks.size());
  size_t residual_idx = 0;
  for (const RigCalibrationObservabilityTrackData& track : group_data.tracks) {
    int track_row_count = 0;
    for (size_t track_residual_idx = 0;
         track_residual_idx < track.num_residual_blocks;
         ++track_residual_idx) {
      track_row_count +=
          add_residual(group_data.track_residual_blocks[residual_idx++]);
    }
    track_row_counts.push_back(track_row_count);
  }
  for (const ceres::ResidualBlockId source_residual :
       group_data.non_track_residual_blocks) {
    add_residual(source_residual);
  }

  ceres::Problem::EvaluateOptions evaluate_options;
  evaluate_options.apply_loss_function = true;
  evaluate_options.num_threads = 1;
  evaluate_options.parameter_blocks = std::move(variable_blocks);
  evaluate_options.residual_blocks = std::move(residual_blocks);
  ceres::CRSMatrix jacobian;
  THROW_CHECK(group_problem.Evaluate(
      evaluate_options, nullptr, nullptr, nullptr, &jacobian));
  return {std::move(jacobian), std::move(track_row_counts)};
}

Eigen::MatrixXd ComputeGroupMarginalInformation(
    const ceres::Problem& source_problem,
    const std::vector<double*>& global_parameter_blocks,
    const RigCalibrationObservabilityGroupData& group_data,
    const double relative_rank_tolerance,
    ceres::Context* context) {
  int num_global_parameters = 0;
  for (const double* parameter_block : global_parameter_blocks) {
    num_global_parameters +=
        source_problem.ParameterBlockTangentSize(parameter_block);
  }
  int num_local_parameters = 0;
  for (const double* parameter_block : group_data.local_pose_blocks) {
    num_local_parameters +=
        source_problem.ParameterBlockTangentSize(parameter_block);
  }
  for (const RigCalibrationObservabilityTrackData& track : group_data.tracks) {
    THROW_CHECK_EQ(source_problem.ParameterBlockTangentSize(track.point_block),
                   3);
  }
  EvaluatedGroupJacobian evaluated = EvaluateGroupJacobian(
      source_problem, global_parameter_blocks, group_data, context);
  return MarginalizeRigCalibrationGroupJacobian(evaluated.jacobian,
                                                num_global_parameters,
                                                num_local_parameters,
                                                evaluated.track_row_counts,
                                                relative_rank_tolerance);
}

}  // namespace

Eigen::MatrixXd MarginalizeRigCalibrationGroupJacobian(
    const ceres::CRSMatrix& jacobian,
    const int num_global_parameters,
    const int num_local_parameters,
    const std::vector<int>& track_row_counts,
    const double relative_rank_tolerance) {
  const int num_calibration_parameters =
      num_global_parameters + num_local_parameters;
  THROW_CHECK_EQ(jacobian.num_cols,
                 num_calibration_parameters + 3 * track_row_counts.size());
  const int total_track_rows =
      std::accumulate(track_row_counts.begin(), track_row_counts.end(), 0);
  THROW_CHECK_LE(total_track_rows, jacobian.num_rows);

  Eigen::MatrixXd lower_information = Eigen::MatrixXd::Zero(
      num_calibration_parameters, num_calibration_parameters);
  Eigen::Matrix<double, Eigen::Dynamic, 3> calibration_point_cross(
      num_calibration_parameters, 3);
  std::vector<uint8_t> active_mask(num_calibration_parameters, false);
  std::vector<int> active_columns;
  active_columns.reserve(num_calibration_parameters);
  std::vector<std::pair<int, double>> calibration_entries;
  calibration_entries.reserve(num_calibration_parameters);

  int row_idx = 0;
  for (size_t track_idx = 0; track_idx < track_row_counts.size(); ++track_idx) {
    active_columns.clear();
    Eigen::Matrix3d point_information = Eigen::Matrix3d::Zero();
    const int point_column_offset = num_calibration_parameters + 3 * track_idx;
    const int num_track_rows = track_row_counts[track_idx];
    for (int track_row_idx = 0; track_row_idx < num_track_rows;
         ++track_row_idx, ++row_idx) {
      calibration_entries.clear();
      Eigen::Vector3d point_jacobian = Eigen::Vector3d::Zero();
      for (int value_idx = jacobian.rows[row_idx];
           value_idx < jacobian.rows[row_idx + 1];
           ++value_idx) {
        const int column = jacobian.cols[value_idx];
        const double value = jacobian.values[value_idx];
        if (column < num_calibration_parameters) {
          calibration_entries.emplace_back(column, value);
          if (active_mask[column] == 0) {
            active_mask[column] = true;
            active_columns.push_back(column);
            calibration_point_cross.row(column).setZero();
          }
        } else {
          point_jacobian[column - point_column_offset] = value;
        }
      }

      AddRowOuterProduct(calibration_entries, &lower_information);
      for (const auto [column, value] : calibration_entries) {
        calibration_point_cross.row(column).noalias() +=
            value * point_jacobian.transpose();
      }
      point_information.noalias() +=
          point_jacobian * point_jacobian.transpose();
    }

    const Eigen::Matrix3d point_information_inverse =
        SymmetricPseudoInverse(point_information, relative_rank_tolerance);
    for (size_t column_idx = 0; column_idx < active_columns.size();
         ++column_idx) {
      const int column1 = active_columns[column_idx];
      const Eigen::Vector3d weighted_cross =
          point_information_inverse *
          calibration_point_cross.row(column1).transpose();
      for (size_t other_idx = 0; other_idx <= column_idx; ++other_idx) {
        const int column2 = active_columns[other_idx];
        lower_information(std::max(column1, column2),
                          std::min(column1, column2)) -=
            calibration_point_cross.row(column2).dot(weighted_cross);
      }
      active_mask[column1] = false;
    }
  }

  for (; row_idx < jacobian.num_rows; ++row_idx) {
    calibration_entries.clear();
    for (int value_idx = jacobian.rows[row_idx];
         value_idx < jacobian.rows[row_idx + 1];
         ++value_idx) {
      if (jacobian.cols[value_idx] < num_calibration_parameters) {
        calibration_entries.emplace_back(jacobian.cols[value_idx],
                                         jacobian.values[value_idx]);
      }
    }
    AddRowOuterProduct(calibration_entries, &lower_information);
  }

  const Eigen::MatrixXd schur =
      lower_information.selfadjointView<Eigen::Lower>();
  const Eigen::MatrixXd global_information =
      schur.topLeftCorner(num_global_parameters, num_global_parameters);
  if (num_local_parameters == 0) {
    return global_information;
  }
  const Eigen::MatrixXd global_local =
      schur.topRightCorner(num_global_parameters, num_local_parameters);
  const Eigen::MatrixXd local_information =
      schur.bottomRightCorner(num_local_parameters, num_local_parameters);
  Eigen::MatrixXd marginal_information =
      global_information -
      global_local *
          SymmetricPseudoInverse(local_information, relative_rank_tolerance) *
          global_local.transpose();
  return 0.5 * (marginal_information + marginal_information.transpose());
}

Eigen::MatrixXd ComputeRigCalibrationMarginalInformation(
    const ceres::Problem& source_problem,
    const std::vector<double*>& global_parameter_blocks,
    const std::vector<RigCalibrationObservabilityGroupData>& groups,
    const double relative_rank_tolerance,
    const int num_threads) {
  THROW_CHECK_GT(groups.size(), 0);
  int num_global_parameters = 0;
  for (const double* parameter_block : global_parameter_blocks) {
    num_global_parameters +=
        source_problem.ParameterBlockTangentSize(parameter_block);
  }
  Eigen::MatrixXd marginal_information =
      Eigen::MatrixXd::Zero(num_global_parameters, num_global_parameters);

  const int effective_num_threads = std::min(
      GetEffectiveNumThreads(num_threads), static_cast<int>(groups.size()));
  std::unique_ptr<ceres::Context> context(ceres::Context::Create());
  ThreadPool thread_pool(effective_num_threads);
  std::vector<std::shared_future<Eigen::MatrixXd>> futures;
  futures.reserve(groups.size());
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    futures.push_back(thread_pool.AddTask([&, group_idx]() {
      return ComputeGroupMarginalInformation(source_problem,
                                             global_parameter_blocks,
                                             groups[group_idx],
                                             relative_rank_tolerance,
                                             context.get());
    }));
  }
  for (std::shared_future<Eigen::MatrixXd>& future : futures) {
    marginal_information += future.get();
  }
  return 0.5 * (marginal_information + marginal_information.transpose());
}

}  // namespace internal
}  // namespace colmap
