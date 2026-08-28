#include "kernel_SensorFromRigLogScale_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SensorFromRigLogScaleRetractKernel(
    double* SensorFromRigLogScale,
    unsigned int SensorFromRigLogScale_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SensorFromRigLogScale_retracted,
    unsigned int out_SensorFromRigLogScale_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(SensorFromRigLogScale,
                                           0 * SensorFromRigLogScale_num_alloc,
                                           global_thread_idx,
                                           r0);
    ReadIdx1<1024, double, double, double>(
        delta, 0 * delta_num_alloc, global_thread_idx, r1);
    r1 = r0 + r1;
    WriteIdx1<1024, double, double, double>(
        out_SensorFromRigLogScale_retracted,
        0 * out_SensorFromRigLogScale_retracted_num_alloc,
        global_thread_idx,
        r1);
  };
}

void SensorFromRigLogScaleRetract(
    double* SensorFromRigLogScale,
    unsigned int SensorFromRigLogScale_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SensorFromRigLogScale_retracted,
    unsigned int out_SensorFromRigLogScale_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SensorFromRigLogScaleRetractKernel<<<n_blocks, 1024>>>(
      SensorFromRigLogScale,
      SensorFromRigLogScale_num_alloc,
      delta,
      delta_num_alloc,
      out_SensorFromRigLogScale_retracted,
      out_SensorFromRigLogScale_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar