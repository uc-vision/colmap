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

#include <algorithm>
#include <vector>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& matrix,
                              const double relative_tolerance) {
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(matrix);
  const double threshold =
      relative_tolerance * std::max(0.0, eig.eigenvalues().maxCoeff());
  Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(matrix.rows());
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    if (eig.eigenvalues()[i] > threshold) {
      inverse_eigenvalues[i] = 1.0 / eig.eigenvalues()[i];
    }
  }
  return eig.eigenvectors() * inverse_eigenvalues.asDiagonal() *
         eig.eigenvectors().transpose();
}

ceres::CRSMatrix ToCRSMatrix(const Eigen::MatrixXd& matrix) {
  ceres::CRSMatrix result;
  result.num_rows = matrix.rows();
  result.num_cols = matrix.cols();
  result.rows.reserve(matrix.rows() + 1);
  result.rows.push_back(0);
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      if (matrix(row, column) != 0.0) {
        result.cols.push_back(column);
        result.values.push_back(matrix(row, column));
      }
    }
    result.rows.push_back(result.values.size());
  }
  return result;
}

TEST(RigCalibrationObservability, MatchesDenseSchurElimination) {
  constexpr int num_global_parameters = 3;
  constexpr int num_local_parameters = 2;
  constexpr int num_calibration_parameters =
      num_global_parameters + num_local_parameters;
  constexpr double rank_tolerance = 1e-10;
  const std::vector<int> track_row_counts = {3, 5};
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(9, 11);

  jacobian(0, 0) = 1.2;
  jacobian(0, 3) = -0.5;
  jacobian(0, 5) = 1.0;
  jacobian(1, 1) = -0.7;
  jacobian(1, 3) = 0.3;
  jacobian(1, 6) = 1.0;
  jacobian(2, 0) = 0.4;
  jacobian(2, 4) = 0.8;
  jacobian(2, 7) = 1.0;
  jacobian(3, 1) = 1.1;
  jacobian(3, 4) = -0.2;
  jacobian(3, 8) = 0.5;
  jacobian(3, 9) = -0.4;

  jacobian(4, 0) = -0.8;
  jacobian(4, 3) = 0.6;
  jacobian(4, 8) = 1.0;
  jacobian(5, 1) = 0.9;
  jacobian(5, 4) = -0.3;
  jacobian(5, 9) = 1.0;
  jacobian(6, 0) = 0.2;
  jacobian(6, 4) = 0.5;
  jacobian(6, 8) = 1.0;
  jacobian(6, 9) = 1.0;
  jacobian(7, 1) = -0.4;
  jacobian(7, 3) = 0.7;
  jacobian(7, 8) = 2.0;
  jacobian(7, 9) = -1.0;

  jacobian(8, 3) = 2.0;
  jacobian(8, 4) = -1.0;

  const Eigen::MatrixXd jacobian_calibration =
      jacobian.leftCols(num_calibration_parameters);
  const Eigen::MatrixXd jacobian_points =
      jacobian.rightCols(3 * track_row_counts.size());
  Eigen::MatrixXd schur =
      jacobian_calibration.transpose() * jacobian_calibration;
  const Eigen::MatrixXd calibration_point =
      jacobian_calibration.transpose() * jacobian_points;
  const Eigen::MatrixXd point_information =
      jacobian_points.transpose() * jacobian_points;
  for (size_t track_idx = 0; track_idx < track_row_counts.size(); ++track_idx) {
    const Eigen::Index offset = 3 * track_idx;
    schur.noalias() -=
        calibration_point.middleCols(offset, 3) *
        PseudoInverse(point_information.block(offset, offset, 3, 3),
                      rank_tolerance) *
        calibration_point.middleCols(offset, 3).transpose();
  }
  schur = 0.5 * (schur + schur.transpose());
  const Eigen::MatrixXd expected =
      schur.topLeftCorner(num_global_parameters, num_global_parameters) -
      schur.topRightCorner(num_global_parameters, num_local_parameters) *
          PseudoInverse(schur.bottomRightCorner(num_local_parameters,
                                                num_local_parameters),
                        rank_tolerance) *
          schur.topRightCorner(num_global_parameters, num_local_parameters)
              .transpose();

  const Eigen::MatrixXd actual =
      internal::MarginalizeRigCalibrationGroupJacobian(ToCRSMatrix(jacobian),
                                                       num_global_parameters,
                                                       num_local_parameters,
                                                       track_row_counts,
                                                       rank_tolerance);

  EXPECT_TRUE(actual.isApprox(expected, 1e-11)) << "expected:\n"
                                                << expected << "\nactual:\n"
                                                << actual;
  EXPECT_TRUE(actual.row(2).isZero());
  EXPECT_TRUE(actual.col(2).isZero());
}

}  // namespace
}  // namespace colmap
