#include "kernel_fixed_rig_sensor_position_prior_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigSensorPositionPriorJtjnjtrDirectKernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        const float* const sensor_from_rig_log_scale_njtr,
        float* sensor_from_rig_log_scale_jac,
        unsigned int sensor_from_rig_log_scale_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_sensor_from_rig_log_scale_njtr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale_njtr, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = r2 * r4;
    ReadIdx3<1024, float, float, float4>(
        sensor_from_rig_log_scale_jac,
        0 * sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r6,
        r7,
        r8);
    r9 = r1 * r4;
    r9 = fmaf(r7, r9, r8 * r5);
    r5 = r4 * r6;
    r9 = fmaf(r0, r5, r9);
    ReadIdx4<1024, float, float, float4>(pose_jac,
                                         4 * pose_jac_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12,
                                         r13);
    r14 = r10 * r4;
    r15 = r11 * r4;
    r15 = fmaf(r8, r15, r7 * r14);
    r15 = fmaf(r3, r5, r15);
    ReadIdx4<1024, float, float, float4>(pose_jac,
                                         8 * pose_jac_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r16,
                                         r17,
                                         r18);
    r19 = r14 * r4;
    r20 = r13 * r4;
    r20 = fmaf(r7, r20, r8 * r19);
    r20 = fmaf(r12, r5, r20);
    r19 = r17 * r4;
    r21 = r18 * r4;
    r21 = fmaf(r8, r21, r7 * r19);
    r21 = fmaf(r16, r5, r21);
    WriteSum4<float, float>((float*)inout_shared, r9, r15, r20, r21);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(pose_jac,
                                         12 * pose_jac_num_alloc,
                                         global_thread_idx,
                                         r21,
                                         r20,
                                         r15,
                                         r9);
    r19 = r20 * r4;
    r22 = r15 * r4;
    r22 = fmaf(r8, r22, r7 * r19);
    r22 = fmaf(r21, r5, r22);
    ReadIdx2<1024, float, float, float2>(
        pose_jac, 16 * pose_jac_num_alloc, global_thread_idx, r19, r23);
    r24 = r19 * r4;
    r25 = r23 * r4;
    r25 = fmaf(r8, r25, r7 * r24);
    r25 = fmaf(r9, r5, r25);
    WriteSum2<float, float>((float*)inout_shared, r22, r25);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<2, float, float>(pose_njtr,
                              4 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r25,
                       r22);
  };
  __syncthreads();
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r5,
                       r24,
                       r26,
                       r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r27, r18, r22 * r23);
    r28 = fmaf(r25, r15, r28);
    r28 = fmaf(r26, r14, r28);
    r28 = fmaf(r24, r11, r28);
    r28 = fmaf(r5, r2, r28);
    r29 = fmaf(r27, r17, r22 * r19);
    r29 = fmaf(r25, r20, r29);
    r29 = fmaf(r26, r13, r29);
    r29 = fmaf(r24, r10, r29);
    r29 = fmaf(r5, r1, r29);
    r29 = fmaf(r7, r29, r8 * r28);
    r21 = fmaf(r25, r21, r22 * r9);
    r21 = fmaf(r26, r12, r21);
    r21 = fmaf(r5, r0, r21);
    r21 = fmaf(r24, r3, r21);
    r21 = fmaf(r27, r16, r21);
    r29 = fmaf(r6, r21, r29);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r29);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
}

void FixedRigSensorPositionPriorJtjnjtrDirect(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    const float* const sensor_from_rig_log_scale_njtr,
    float* sensor_from_rig_log_scale_jac,
    unsigned int sensor_from_rig_log_scale_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_sensor_from_rig_log_scale_njtr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigSensorPositionPriorJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      sensor_from_rig_log_scale_njtr,
      sensor_from_rig_log_scale_jac,
      sensor_from_rig_log_scale_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_sensor_from_rig_log_scale_njtr,
      problem_size);
}

}  // namespace caspar