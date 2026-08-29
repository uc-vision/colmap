#include "kernel_fixed_rig_log_scale_prior_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigLogScalePriorResJacFirstKernel(
        const float* const sensor_from_rig_log_scale,
        const float* const target,
        const float* const sqrt_information,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_sensor_from_rig_log_scale_njtr,
        float* const out_sensor_from_rig_log_scale_precond_diag,
        float* const out_sensor_from_rig_log_scale_precond_tril,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_rTr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_precond_diag_local[1];

  float r0, r1, r2, r3;
  LoadUnique<1, float, float>(sqrt_information, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r0);
  };
  __syncthreads();
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r1);
  };
  __syncthreads();
  LoadUnique<1, float, float>(target, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r3, r1);
    r1 = r0 * r2;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r1);
    r1 = r0 * r1;
    r2 = r2 * r1;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  if (global_thread_idx < problem_size) {
    r1 = r3 * r1;
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    r0 = r0 * r0;
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r0);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                       out_sensor_from_rig_log_scale_precond_diag,
                       1);
}

void FixedRigLogScalePriorResJacFirst(
    const float* const sensor_from_rig_log_scale,
    const float* const target,
    const float* const sqrt_information,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_sensor_from_rig_log_scale_njtr,
    float* const out_sensor_from_rig_log_scale_precond_diag,
    float* const out_sensor_from_rig_log_scale_precond_tril,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigLogScalePriorResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig_log_scale,
      target,
      sqrt_information,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_sensor_from_rig_log_scale_njtr,
      out_sensor_from_rig_log_scale_precond_diag,
      out_sensor_from_rig_log_scale_precond_tril,
      problem_size);
}

}  // namespace caspar