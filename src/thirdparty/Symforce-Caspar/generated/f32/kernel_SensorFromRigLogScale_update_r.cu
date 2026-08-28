#include "kernel_SensorFromRigLogScale_update_r.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SensorFromRigLogScaleUpdateRKernel(
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    const float* const negalpha,
    float* out_SensorFromRigLogScale_r_kp1,
    unsigned int out_SensorFromRigLogScale_r_kp1_num_alloc,
    float* const out_SensorFromRigLogScale_r_kp1_norm2_tot,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_SensorFromRigLogScale_r_kp1_norm2_tot_local[1];

  float r0, r1, r2;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(SensorFromRigLogScale_r_k,
                                        0 * SensorFromRigLogScale_r_k_num_alloc,
                                        global_thread_idx,
                                        r0);
    ReadIdx1<1024, float, float, float>(SensorFromRigLogScale_w,
                                        0 * SensorFromRigLogScale_w_num_alloc,
                                        global_thread_idx,
                                        r1);
  };
  LoadUnique<1, float, float>(negalpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r1, r2, r0);
    WriteIdx1<1024, float, float, float>(
        out_SensorFromRigLogScale_r_kp1,
        0 * out_SensorFromRigLogScale_r_kp1_num_alloc,
        global_thread_idx,
        r2);
    r2 = r2 * r2;
  };
  SumStore<float>(out_SensorFromRigLogScale_r_kp1_norm2_tot_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  SumFlushFinal<float>(out_SensorFromRigLogScale_r_kp1_norm2_tot_local,
                       out_SensorFromRigLogScale_r_kp1_norm2_tot,
                       1);
}

void SensorFromRigLogScaleUpdateR(
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    const float* const negalpha,
    float* out_SensorFromRigLogScale_r_kp1,
    unsigned int out_SensorFromRigLogScale_r_kp1_num_alloc,
    float* const out_SensorFromRigLogScale_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SensorFromRigLogScaleUpdateRKernel<<<n_blocks, 1024>>>(
      SensorFromRigLogScale_r_k,
      SensorFromRigLogScale_r_k_num_alloc,
      SensorFromRigLogScale_w,
      SensorFromRigLogScale_w_num_alloc,
      negalpha,
      out_SensorFromRigLogScale_r_kp1,
      out_SensorFromRigLogScale_r_kp1_num_alloc,
      out_SensorFromRigLogScale_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar