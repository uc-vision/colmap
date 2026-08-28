#include "kernel_SensorFromRigLogScale_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SensorFromRigLogScaleAlphaNumeratorDenominatorKernel(
        double* SensorFromRigLogScale_p_kp1,
        unsigned int SensorFromRigLogScale_p_kp1_num_alloc,
        double* SensorFromRigLogScale_r_k,
        unsigned int SensorFromRigLogScale_r_k_num_alloc,
        double* SensorFromRigLogScale_w,
        unsigned int SensorFromRigLogScale_w_num_alloc,
        double* const SensorFromRigLogScale_total_ag,
        double* const SensorFromRigLogScale_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SensorFromRigLogScale_total_ag_local[1];

  __shared__ double SensorFromRigLogScale_total_ac_local[1];

  double r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        SensorFromRigLogScale_p_kp1,
        0 * SensorFromRigLogScale_p_kp1_num_alloc,
        global_thread_idx,
        r0);
    ReadIdx1<1024, double, double, double>(
        SensorFromRigLogScale_r_k,
        0 * SensorFromRigLogScale_r_k_num_alloc,
        global_thread_idx,
        r1);
    r1 = r0 * r1;
  };
  SumStore<double>(SensorFromRigLogScale_total_ag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        SensorFromRigLogScale_w,
        0 * SensorFromRigLogScale_w_num_alloc,
        global_thread_idx,
        r1);
    r1 = r0 * r1;
  };
  SumStore<double>(SensorFromRigLogScale_total_ac_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  SumFlushFinal<double>(
      SensorFromRigLogScale_total_ag_local, SensorFromRigLogScale_total_ag, 1);
  SumFlushFinal<double>(
      SensorFromRigLogScale_total_ac_local, SensorFromRigLogScale_total_ac, 1);
}

void SensorFromRigLogScaleAlphaNumeratorDenominator(
    double* SensorFromRigLogScale_p_kp1,
    unsigned int SensorFromRigLogScale_p_kp1_num_alloc,
    double* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    double* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    double* const SensorFromRigLogScale_total_ag,
    double* const SensorFromRigLogScale_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SensorFromRigLogScaleAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      SensorFromRigLogScale_p_kp1,
      SensorFromRigLogScale_p_kp1_num_alloc,
      SensorFromRigLogScale_r_k,
      SensorFromRigLogScale_r_k_num_alloc,
      SensorFromRigLogScale_w,
      SensorFromRigLogScale_w_num_alloc,
      SensorFromRigLogScale_total_ag,
      SensorFromRigLogScale_total_ac,
      problem_size);
}

}  // namespace caspar