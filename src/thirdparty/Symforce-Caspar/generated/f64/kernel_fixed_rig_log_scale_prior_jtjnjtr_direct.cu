#include "kernel_fixed_rig_log_scale_prior_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigLogScalePriorJtjnjtrDirectKernel(
        const double* const sensor_from_rig_log_scale_njtr,
        double* sensor_from_rig_log_scale_jac,
        unsigned int sensor_from_rig_log_scale_jac_num_alloc,
        double* const out_sensor_from_rig_log_scale_njtr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_sensor_from_rig_log_scale_njtr_local[1];
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_njtr_local,
                        out_sensor_from_rig_log_scale_njtr,
                        1);
}

void FixedRigLogScalePriorJtjnjtrDirect(
    const double* const sensor_from_rig_log_scale_njtr,
    double* sensor_from_rig_log_scale_jac,
    unsigned int sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigLogScalePriorJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      sensor_from_rig_log_scale_njtr,
      sensor_from_rig_log_scale_jac,
      sensor_from_rig_log_scale_jac_num_alloc,
      out_sensor_from_rig_log_scale_njtr,
      problem_size);
}

}  // namespace caspar