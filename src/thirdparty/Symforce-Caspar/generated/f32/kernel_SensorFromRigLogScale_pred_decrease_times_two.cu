#include "kernel_SensorFromRigLogScale_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SensorFromRigLogScalePredDecreaseTimesTwoKernel(
        float* SensorFromRigLogScale_step,
        unsigned int SensorFromRigLogScale_step_num_alloc,
        float* SensorFromRigLogScale_precond_diag,
        unsigned int SensorFromRigLogScale_precond_diag_num_alloc,
        const float* const diag,
        float* SensorFromRigLogScale_njtr,
        unsigned int SensorFromRigLogScale_njtr_num_alloc,
        float* const out_SensorFromRigLogScale_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_SensorFromRigLogScale_pred_dec_local[1];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        SensorFromRigLogScale_step,
        0 * SensorFromRigLogScale_step_num_alloc,
        global_thread_idx,
        r0);
    ReadIdx1<1024, float, float, float>(
        SensorFromRigLogScale_njtr,
        0 * SensorFromRigLogScale_njtr_num_alloc,
        global_thread_idx,
        r1);
    ReadIdx1<1024, float, float, float>(
        SensorFromRigLogScale_precond_diag,
        0 * SensorFromRigLogScale_precond_diag_num_alloc,
        global_thread_idx,
        r2);
    r3 = r0 * r2;
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r4, r3, r1);
    r3 = r0 * r3;
  };
  SumStore<float>(out_SensorFromRigLogScale_pred_dec_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  SumFlushFinal<float>(out_SensorFromRigLogScale_pred_dec_local,
                       out_SensorFromRigLogScale_pred_dec,
                       1);
}

void SensorFromRigLogScalePredDecreaseTimesTwo(
    float* SensorFromRigLogScale_step,
    unsigned int SensorFromRigLogScale_step_num_alloc,
    float* SensorFromRigLogScale_precond_diag,
    unsigned int SensorFromRigLogScale_precond_diag_num_alloc,
    const float* const diag,
    float* SensorFromRigLogScale_njtr,
    unsigned int SensorFromRigLogScale_njtr_num_alloc,
    float* const out_SensorFromRigLogScale_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SensorFromRigLogScalePredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      SensorFromRigLogScale_step,
      SensorFromRigLogScale_step_num_alloc,
      SensorFromRigLogScale_precond_diag,
      SensorFromRigLogScale_precond_diag_num_alloc,
      diag,
      SensorFromRigLogScale_njtr,
      SensorFromRigLogScale_njtr_num_alloc,
      out_SensorFromRigLogScale_pred_dec,
      problem_size);
}

}  // namespace caspar