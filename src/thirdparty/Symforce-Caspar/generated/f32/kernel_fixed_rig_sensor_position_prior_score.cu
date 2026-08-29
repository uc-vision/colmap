#include "kernel_fixed_rig_sensor_position_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigSensorPositionPriorScoreKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        const float* const sensor_from_rig_log_scale,
        float* position,
        unsigned int position_num_alloc,
        float* sqrt_information,
        unsigned int sqrt_information_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sqrt_information,
                                         0 * sqrt_information_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
    r4 = -1.00000000000000000e+00;
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10,
                                         r11);
    r12 = r8 * r10;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = -2.00000000000000000e+00;
    r15 = r9 * r11;
    r16 = fmaf(r14, r15, r12);
    r17 = r9 * r9;
    r17 = r17 * r14;
    r18 = 1.00000000000000000e+00;
    r19 = r8 * r8;
    r19 = fmaf(r14, r19, r18);
    r20 = r17 + r19;
    r20 = fmaf(r7, r20, r5 * r16);
    r16 = r9 * r10;
    r16 = r16 * r13;
    r21 = r8 * r11;
    r21 = fmaf(r13, r21, r16);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r22,
                                         r23,
                                         r24);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r26 = 2.71828182845904523536;
    r25 = powf(r26, r25);
    r20 = fmaf(r6, r21, r20);
    r20 = fmaf(r24, r25, r20);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r24,
                       r21,
                       r26,
                       r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r27, r8, r24 * r11);
    r29 = r21 * r10;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r26, r9, r28);
    r29 = fmaf(r21, r8, r26 * r11);
    r30 = r24 * r9;
    r29 = fmaf(r4, r30, r29);
    r29 = fmaf(r27, r10, r29);
    r30 = r13 * r29;
    r31 = r28 * r30;
    r32 = r26 * r8;
    r32 = fmaf(r4, r32, r21 * r11);
    r32 = fmaf(r27, r9, r32);
    r32 = fmaf(r24, r10, r32);
    r33 = fmaf(r21, r9, r24 * r8);
    r33 = fmaf(r26, r10, r33);
    r33 = fmaf(r4, r33, r27 * r11);
    r27 = r14 * r33;
    r34 = fmaf(r32, r27, r31);
    r15 = fmaf(r13, r15, r12);
    r12 = r10 * r11;
    r35 = r8 * r9;
    r35 = r35 * r13;
    r12 = fmaf(r14, r12, r35);
    r12 = fmaf(r6, r12, r7 * r15);
    r17 = r18 + r17;
    r15 = r10 * r10;
    r15 = r14 * r15;
    r17 = r17 + r15;
    r12 = fmaf(r5, r17, r12);
    r12 = fmaf(r22, r25, r12);
    r22 = r29 * r29;
    r22 = r22 * r14;
    r17 = r18 + r22;
    r36 = r32 * r32;
    r36 = r14 * r36;
    r17 = r17 + r36;
    r17 = fmaf(r17, r12, r34 * r20);
    r34 = r10 * r11;
    r34 = fmaf(r13, r34, r35);
    r19 = r15 + r19;
    r19 = fmaf(r6, r19, r5 * r34);
    r6 = r8 * r11;
    r6 = fmaf(r14, r6, r16);
    r19 = fmaf(r7, r6, r19);
    r19 = fmaf(r23, r25, r19);
    r25 = r13 * r28;
    r25 = r25 * r32;
    r23 = fmaf(r33, r30, r25);
    r17 = fmaf(r23, r19, r17);
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r23, r6, r7);
    r23 = fmaf(r23, r4, r4 * r17);
    ReadIdx1<1024, float, float, float>(sqrt_information,
                                        8 * sqrt_information_num_alloc,
                                        global_thread_idx,
                                        r17);
    r16 = r13 * r32;
    r16 = fmaf(r33, r16, r31);
    r30 = r32 * r30;
    r31 = fmaf(r28, r27, r30);
    r31 = fmaf(r19, r31, r12 * r16);
    r36 = r18 + r36;
    r16 = r28 * r28;
    r16 = r14 * r16;
    r36 = r36 + r16;
    r31 = fmaf(r20, r36, r31);
    r31 = fmaf(r4, r31, r7 * r4);
    r17 = fmaf(r17, r31, r2 * r23);
    ReadIdx4<1024, float, float, float4>(sqrt_information,
                                         4 * sqrt_information_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r7,
                                         r36,
                                         r14);
    r34 = r13 * r28;
    r34 = fmaf(r33, r34, r30);
    r27 = fmaf(r29, r27, r25);
    r27 = fmaf(r12, r27, r20 * r34);
    r22 = r18 + r22;
    r22 = r22 + r16;
    r27 = fmaf(r19, r22, r27);
    r4 = fmaf(r6, r4, r4 * r27);
    r17 = fmaf(r7, r4, r17);
    r36 = fmaf(r36, r31, r0 * r23);
    r36 = fmaf(r3, r4, r36);
    r36 = fmaf(r36, r36, r17 * r17);
    r31 = fmaf(r14, r31, r1 * r23);
    r31 = fmaf(r2, r4, r31);
    r36 = fmaf(r31, r31, r36);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r36);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void FixedRigSensorPositionPriorScore(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* position,
    unsigned int position_num_alloc,
    float* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigSensorPositionPriorScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      sensor_from_rig_log_scale,
      position,
      position_num_alloc,
      sqrt_information,
      sqrt_information_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar