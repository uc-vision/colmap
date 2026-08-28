#include "kernel_SensorFromRigLogScale_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SensorFromRigLogScaleStartWContributeKernel(
        double* SensorFromRigLogScale_precond_diag,
        unsigned int SensorFromRigLogScale_precond_diag_num_alloc,
        const double* const diag,
        double* SensorFromRigLogScale_p,
        unsigned int SensorFromRigLogScale_p_num_alloc,
        double* out_SensorFromRigLogScale_w,
        unsigned int out_SensorFromRigLogScale_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        SensorFromRigLogScale_precond_diag,
        0 * SensorFromRigLogScale_precond_diag_num_alloc,
        global_thread_idx,
        r0);
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r0 * r1;
    ReadIdx1<1024, double, double, double>(
        SensorFromRigLogScale_p,
        0 * SensorFromRigLogScale_p_num_alloc,
        global_thread_idx,
        r0);
    r1 = r1 * r0;
    AddIdx1<1024, double, double, double>(
        out_SensorFromRigLogScale_w,
        0 * out_SensorFromRigLogScale_w_num_alloc,
        global_thread_idx,
        r1);
  };
}

void SensorFromRigLogScaleStartWContribute(
    double* SensorFromRigLogScale_precond_diag,
    unsigned int SensorFromRigLogScale_precond_diag_num_alloc,
    const double* const diag,
    double* SensorFromRigLogScale_p,
    unsigned int SensorFromRigLogScale_p_num_alloc,
    double* out_SensorFromRigLogScale_w,
    unsigned int out_SensorFromRigLogScale_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SensorFromRigLogScaleStartWContributeKernel<<<n_blocks, 1024>>>(
      SensorFromRigLogScale_precond_diag,
      SensorFromRigLogScale_precond_diag_num_alloc,
      diag,
      SensorFromRigLogScale_p,
      SensorFromRigLogScale_p_num_alloc,
      out_SensorFromRigLogScale_w,
      out_SensorFromRigLogScale_w_num_alloc,
      problem_size);
}

}  // namespace caspar