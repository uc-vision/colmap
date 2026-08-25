#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "solver.h"
#include "solver_tools.h"
#include <cuda_runtime.h>

namespace caspar {

void GraphSolver::SetRigSchurTopology(
    const std::vector<unsigned int>& pose_indices,
    const std::vector<unsigned int>& point_indices,
    const std::vector<unsigned int>& fixed_pose_point_indices,
    const std::vector<unsigned int>& fixed_point_pose_indices) {
  if (pose_indices.size() !=
          pinhole_split_fixed_focal_fixed_principal_point_num_ ||
      point_indices.size() !=
          pinhole_split_fixed_focal_fixed_principal_point_num_ ||
      fixed_pose_point_indices.size() !=
          pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_ ||
      fixed_point_pose_indices.size() !=
          pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_) {
    throw std::invalid_argument(
        "Rig Schur topology does not match generated factor counts");
  }
  rig_schur_solver_ =
      std::make_unique<RigSchurSolver>(device_id_,
                                       PinholePose_num_,
                                       Point_num_,
                                       pose_indices,
                                       point_indices,
                                       fixed_pose_point_indices,
                                       fixed_point_pose_indices);
}

SolveResult GraphSolver::solve_rig_schur(bool print_progress,
                                         bool verbose_logging) {
  if (!rig_schur_solver_) {
    throw std::runtime_error("Rig Schur topology has not been set");
  }

  cudaSetDevice(device_id_);
  SolveResult result;
  result.exit_reason = ExitReason::MAX_ITERATIONS;
  result.iteration_count = 0;
  float diag = params_.diag_init;
  cudaMemcpy(
      solver__current_diag_, &diag, sizeof(float), cudaMemcpyHostToDevice);
  float up_scale = params_.diag_scaling_up;

  const auto t0 = std::chrono::steady_clock::now();
  auto t_prev = t0;
  float score_best = DoResJacFirst();
  result.initial_score = score_best;
  bool needs_linearization = false;
  if (print_progress) {
    printf("                                 score_init: % .6e\n", score_best);
  }

  for (solver_iter_ = 0; solver_iter_ < params_.solver_iter_max;
       ++solver_iter_) {
    if (needs_linearization) {
      DoResJac();
      needs_linearization = false;
    }

    Zero(nodes__PinholeCalib__step_,
         nodes__SimpleRadialPrincipalPoint__step_end__);
    const bool solved = rig_schur_solver_->SolveStep(
        diag,
        facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_,
        facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_,
        nodes__PinholePose__r_0_,
        nodes__PinholePose__precond_diag_,
        nodes__PinholePose__precond_tril_,
        nodes__Point__r_0_,
        nodes__Point__precond_diag_,
        nodes__Point__precond_tril_,
        nodes__PinholePose__step_,
        nodes__Point__step_);

    float score_current = score_best;
    float quality = 0.0f;
    bool step_accepted = false;
    if (solved) {
      score_current = DoRetractScore();
    }

    const float diag_current = diag;
    bool reached_diag_exit = false;
    if (solved &&
        score_current < score_best * params_.solver_rel_decrease_min) {
      quality = (score_best - score_current) / GetPredDecrease();
      const float quality_tmp = 2 * quality - 1;
      const float scale =
          std::max(params_.diag_scaling_down,
                   1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(float), cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_current;
      step_accepted = true;
      needs_linearization = true;
      std::swap(nodes__PinholeCalib__storage_current_,
                nodes__PinholeCalib__storage_check_);
      std::swap(nodes__PinholeFocal__storage_current_,
                nodes__PinholeFocal__storage_check_);
      std::swap(nodes__PinholePose__storage_current_,
                nodes__PinholePose__storage_check_);
      std::swap(nodes__PinholePrincipalPoint__storage_current_,
                nodes__PinholePrincipalPoint__storage_check_);
      std::swap(nodes__Point__storage_current_, nodes__Point__storage_check_);
      std::swap(nodes__SimpleRadialCalib__storage_current_,
                nodes__SimpleRadialCalib__storage_check_);
      std::swap(nodes__SimpleRadialFocalAndExtra__storage_current_,
                nodes__SimpleRadialFocalAndExtra__storage_check_);
      std::swap(nodes__SimpleRadialPose__storage_current_,
                nodes__SimpleRadialPose__storage_check_);
      std::swap(nodes__SimpleRadialPrincipalPoint__storage_current_,
                nodes__SimpleRadialPrincipalPoint__storage_check_);
    } else {
      diag *= up_scale;
      reached_diag_exit = diag > params_.diag_exit_value;
      if (!reached_diag_exit) {
        cudaMemcpy(solver__current_diag_,
                   &diag,
                   sizeof(float),
                   cudaMemcpyHostToDevice);
        up_scale *= 2;
      }
    }

    const auto t_now = std::chrono::steady_clock::now();
    const double dt_inc = std::chrono::duration<double>(t_now - t_prev).count();
    const double dt_tot = std::chrono::duration<double>(t_now - t0).count();
    if (verbose_logging) {
      result.iterations.push_back({solver_iter_,
                                   0,
                                   score_current,
                                   score_best,
                                   quality,
                                   diag_current,
                                   dt_inc,
                                   dt_tot,
                                   step_accepted});
    }
    if (print_progress) {
      printf("solver_iter: % 3d  ", solver_iter_);
      printf("score_current: % 13.6e  ", score_current);
      printf("score_best: % 13.6e  ", score_best);
      printf("step_quality: % 7.3f  ", quality);
      printf("diag: % 6.3e  ", diag_current);
      printf("dt_inc: % 10.6f  ", dt_inc);
      printf("dt_tot: % 10.6f\n", dt_tot);
    }
    t_prev = t_now;

    result.iteration_count = solver_iter_ + 1;
    if (reached_diag_exit) {
      result.exit_reason = ExitReason::CONVERGED_DIAG_EXIT;
      break;
    }
    if (score_best <= params_.score_exit_value) {
      result.exit_reason = ExitReason::CONVERGED_SCORE_THRESHOLD;
      break;
    }
  }

  const auto t_final = std::chrono::steady_clock::now();
  result.final_score = score_best;
  result.runtime = std::chrono::duration<double>(t_final - t0).count();
  return result;
}

}  // namespace caspar
