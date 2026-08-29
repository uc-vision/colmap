#include "kernel_fixed_rig_pinhole_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigPinholeScoreKernel(float* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               float* sensor_from_rig,
                               unsigned int sensor_from_rig_num_alloc,
                               const float* const sensor_from_rig_log_scale,
                               float* calib,
                               unsigned int calib_num_alloc,
                               float* point,
                               unsigned int point_num_alloc,
                               SharedIndex* point_indices,
                               float* pixel,
                               unsigned int pixel_num_alloc,
                               const float* const reprojection_loss_scale,
                               float* const out_rTr,
                               size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r5 = fmaf(r5, r6, r3);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r3,
                       r7,
                       r8);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r9,
                       r10,
                       r11,
                       r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
    r17 = fmaf(r12, r13, r9 * r16);
    r18 = r10 * r15;
    r17 = fmaf(r6, r18, r17);
    r17 = fmaf(r11, r14, r17);
    r18 = 2.00000000000000000e+00;
    r19 = r11 * r13;
    r19 = fmaf(r6, r19, r10 * r16);
    r19 = fmaf(r12, r14, r19);
    r19 = fmaf(r9, r15, r19);
    r20 = r18 * r19;
    r21 = r17 * r20;
    r22 = fmaf(r10, r13, r11 * r16);
    r23 = r9 * r14;
    r22 = fmaf(r6, r23, r22);
    r22 = fmaf(r12, r15, r22);
    r23 = fmaf(r10, r14, r9 * r13);
    r23 = fmaf(r11, r15, r23);
    r23 = fmaf(r6, r23, r12 * r16);
    r12 = r22 * r23;
    r24 = fmaf(r18, r12, r21);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r25,
                       r26,
                       r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r15 * r16;
    r29 = r13 * r14;
    r29 = r29 * r18;
    r28 = fmaf(r18, r28, r29);
    r28 = fmaf(r25, r28, r3 * r24);
    r24 = -2.00000000000000000e+00;
    r30 = r15 * r15;
    r30 = r24 * r30;
    r31 = 1.00000000000000000e+00;
    r32 = r13 * r13;
    r32 = fmaf(r24, r32, r31);
    r33 = r30 + r32;
    r34 = r14 * r15;
    r34 = r34 * r18;
    r35 = r16 * r24;
    r36 = fmaf(r13, r35, r34);
    r37 = r17 * r23;
    r38 = r22 * r20;
    r37 = fmaf(r24, r37, r38);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r39,
                                         r40,
                                         r41);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = 2.71828182845904523536;
    r42 = powf(r43, r42);
    r43 = r22 * r22;
    r43 = r43 * r24;
    r44 = r31 + r43;
    r45 = r17 * r17;
    r45 = r24 * r45;
    r44 = r44 + r45;
    r28 = fmaf(r26, r33, r28);
    r28 = fmaf(r27, r36, r28);
    r28 = fmaf(r8, r37, r28);
    r28 = fmaf(r40, r42, r28);
    r28 = fmaf(r7, r44, r28);
    r44 = r1 * r28;
    r40 = 9.99999999999999955e-07;
    r37 = r18 * r17;
    r37 = r37 * r22;
    r22 = r19 * r23;
    r22 = fmaf(r24, r22, r37);
    r36 = r13 * r15;
    r36 = r36 * r18;
    r33 = fmaf(r14, r35, r36);
    r33 = fmaf(r25, r33, r3 * r22);
    r22 = r14 * r14;
    r22 = r24 * r22;
    r32 = r22 + r32;
    r46 = r13 * r16;
    r46 = fmaf(r18, r46, r34);
    r34 = r18 * r17;
    r34 = fmaf(r23, r34, r38);
    r45 = r31 + r45;
    r38 = r19 * r19;
    r38 = r38 * r24;
    r45 = r45 + r38;
    r33 = fmaf(r27, r32, r33);
    r33 = fmaf(r26, r46, r33);
    r33 = fmaf(r7, r34, r33);
    r33 = fmaf(r41, r42, r33);
    r33 = fmaf(r8, r45, r33);
    r45 = copysign(1.0, r33);
    r45 = fmaf(r40, r45, r33);
    r45 = 1.0 / r45;
    r5 = fmaf(r45, r44, r5);
    r5 = r5 * r5;
  };
  LoadUnique<1, float, float>(reprojection_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r44);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = r44 * r44;
    r44 = 1.0 / r44;
    r6 = fmaf(r4, r6, r2);
    r43 = r31 + r43;
    r43 = r43 + r38;
    r12 = fmaf(r24, r12, r21);
    r12 = fmaf(r7, r12, r3 * r43);
    r20 = fmaf(r23, r20, r37);
    r37 = r14 * r16;
    r37 = fmaf(r18, r37, r36);
    r35 = fmaf(r15, r35, r29);
    r30 = r31 + r30;
    r30 = r30 + r22;
    r12 = fmaf(r8, r20, r12);
    r12 = fmaf(r27, r37, r12);
    r12 = fmaf(r26, r35, r12);
    r12 = fmaf(r25, r30, r12);
    r12 = fmaf(r39, r42, r12);
    r42 = r0 * r12;
    r6 = fmaf(r45, r42, r6);
    r6 = r6 * r6;
    r42 = r5 + r6;
    r44 = fmaf(r42, r44, r31);
    r44 = sqrtf(r44);
    r44 = r31 + r44;
    r44 = 1.0 / r44;
    r44 = r18 * r44;
    r44 = fmaf(r6, r44, r5 * r44);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r44);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void FixedRigPinholeScore(float* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          float* sensor_from_rig,
                          unsigned int sensor_from_rig_num_alloc,
                          const float* const sensor_from_rig_log_scale,
                          float* calib,
                          unsigned int calib_num_alloc,
                          float* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          float* pixel,
                          unsigned int pixel_num_alloc,
                          const float* const reprojection_loss_scale,
                          float* const out_rTr,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigPinholeScoreKernel<<<n_blocks, 1024>>>(pose,
                                                 pose_num_alloc,
                                                 pose_indices,
                                                 sensor_from_rig,
                                                 sensor_from_rig_num_alloc,
                                                 sensor_from_rig_log_scale,
                                                 calib,
                                                 calib_num_alloc,
                                                 point,
                                                 point_num_alloc,
                                                 point_indices,
                                                 pixel,
                                                 pixel_num_alloc,
                                                 reprojection_loss_scale,
                                                 out_rTr,
                                                 problem_size);
}

}  // namespace caspar