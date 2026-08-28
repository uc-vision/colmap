#include "kernel_SensorFromRigLogScale_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SensorFromRigLogScaleUpdateMpKernel(
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_Mp,
    unsigned int SensorFromRigLogScale_Mp_num_alloc,
    const float* const beta,
    float* out_SensorFromRigLogScale_Mp_kp1,
    unsigned int out_SensorFromRigLogScale_Mp_kp1_num_alloc,
    float* out_SensorFromRigLogScale_w,
    unsigned int out_SensorFromRigLogScale_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(SensorFromRigLogScale_Mp,
                                        0 * SensorFromRigLogScale_Mp_num_alloc,
                                        global_thread_idx,
                                        r0);
    ReadIdx1<1024, float, float, float>(SensorFromRigLogScale_r_k,
                                        0 * SensorFromRigLogScale_r_k_num_alloc,
                                        global_thread_idx,
                                        r1);
  };
  LoadUnique<1, float, float>(beta, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r0, r2, r1);
    WriteIdx1<1024, float, float, float>(
        out_SensorFromRigLogScale_Mp_kp1,
        0 * out_SensorFromRigLogScale_Mp_kp1_num_alloc,
        global_thread_idx,
        r2);
    WriteIdx1<1024, float, float, float>(
        out_SensorFromRigLogScale_w,
        0 * out_SensorFromRigLogScale_w_num_alloc,
        global_thread_idx,
        r2);
  };
}

void SensorFromRigLogScaleUpdateMp(
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_Mp,
    unsigned int SensorFromRigLogScale_Mp_num_alloc,
    const float* const beta,
    float* out_SensorFromRigLogScale_Mp_kp1,
    unsigned int out_SensorFromRigLogScale_Mp_kp1_num_alloc,
    float* out_SensorFromRigLogScale_w,
    unsigned int out_SensorFromRigLogScale_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SensorFromRigLogScaleUpdateMpKernel<<<n_blocks, 1024>>>(
      SensorFromRigLogScale_r_k,
      SensorFromRigLogScale_r_k_num_alloc,
      SensorFromRigLogScale_Mp,
      SensorFromRigLogScale_Mp_num_alloc,
      beta,
      out_SensorFromRigLogScale_Mp_kp1,
      out_SensorFromRigLogScale_Mp_kp1_num_alloc,
      out_SensorFromRigLogScale_w,
      out_SensorFromRigLogScale_w_num_alloc,
      problem_size);
}

}  // namespace caspar