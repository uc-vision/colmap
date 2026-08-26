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

#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/crs_matrix.h>

namespace colmap {
namespace internal {

struct RigCalibrationObservabilityTrackData {
  RigCalibrationObservabilityTrackData(double* point_block,
                                       size_t num_residual_blocks)
      : point_block(point_block), num_residual_blocks(num_residual_blocks) {}

  double* point_block;
  size_t num_residual_blocks;
};

struct RigCalibrationObservabilityGroupData {
  std::vector<RigCalibrationObservabilityTrackData> tracks;
  std::vector<ceres::ResidualBlockId> track_residual_blocks;
  std::vector<double*> local_pose_blocks;
  std::vector<ceres::ResidualBlockId> non_track_residual_blocks;
};

// Computes the shared-parameter information after independently eliminating
// every group's point and local-pose parameters. Groups are evaluated in
// parallel and accumulated in input order.
Eigen::MatrixXd ComputeRigCalibrationMarginalInformation(
    const ceres::Problem& source_problem,
    const std::vector<double*>& global_parameter_blocks,
    const std::vector<RigCalibrationObservabilityGroupData>& groups,
    double relative_rank_tolerance,
    int num_threads);

Eigen::MatrixXd MarginalizeRigCalibrationGroupJacobian(
    const ceres::CRSMatrix& jacobian,
    int num_global_parameters,
    int num_local_parameters,
    const std::vector<int>& track_row_counts,
    double relative_rank_tolerance);

}  // namespace internal
}  // namespace colmap
