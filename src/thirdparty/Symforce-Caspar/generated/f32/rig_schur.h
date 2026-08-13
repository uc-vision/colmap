#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace caspar {

class RigSchurSolver {
 public:
  RigSchurSolver(int device_id,
                 size_t pose_num,
                 size_t point_num,
                 const std::vector<unsigned int>& pose_indices,
                 const std::vector<unsigned int>& point_indices,
                 const std::vector<unsigned int>& fixed_pose_point_indices,
                 const std::vector<unsigned int>& fixed_point_pose_indices);
  ~RigSchurSolver();

  RigSchurSolver(const RigSchurSolver&) = delete;
  RigSchurSolver& operator=(const RigSchurSolver&) = delete;

  bool SolveStep(float diag,
                 const float* pose_jac,
                 const float* point_jac,
                 const float* pose_rhs,
                 const float* pose_diag,
                 const float* pose_tril,
                 const float* point_rhs,
                 const float* point_diag,
                 const float* point_tril,
                 float* pose_step,
                 float* point_step);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace caspar
