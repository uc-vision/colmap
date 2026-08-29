#include "kernel_fixed_rig_log_scale_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedRigLogScalePriorScoreKernel(
    const double* const sensor_from_rig_log_scale,
    const double* const target,
    const double* const sqrt_information,
    double* const out_rTr,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_rTr_local[1];

  double r0, r1, r2;
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r0);
  };
  __syncthreads();
  LoadUnique<1, double, double>(target, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = -1.00000000000000000e+00;
    r2 = fma(r1, r2, r0);
    r2 = r2 * r2;
  };
  LoadUnique<1, double, double>(sqrt_information, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r1 * r1;
    r1 = r2 * r1;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedRigLogScalePriorScore(const double* const sensor_from_rig_log_scale,
                                const double* const target,
                                const double* const sqrt_information,
                                double* const out_rTr,
                                size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigLogScalePriorScoreKernel<<<n_blocks, 1024>>>(
      sensor_from_rig_log_scale,
      target,
      sqrt_information,
      out_rTr,
      problem_size);
}

}  // namespace caspar