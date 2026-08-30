#include "kernel_row_fixed_rig_pinhole_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RowFixedRigPinholeScoreKernel(float* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  float* sensor_calibration,
                                  unsigned int sensor_calibration_num_alloc,
                                  SharedIndex* sensor_calibration_indices,
                                  const float* const sensor_from_rig_log_scale,
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
  __shared__ SharedIndex sensor_calibration_indices_loc[1024];
  sensor_calibration_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? sensor_calibration_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;
  LoadShared<3, float, float>(sensor_calibration,
                              8 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r1);
  };
  LoadShared<4, float, float>(sensor_calibration,
                              4 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r1,
                       r6,
                       r7,
                       r8);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r12,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  LoadShared<4, float, float>(sensor_calibration,
                              0 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r16,
                       r17,
                       r18,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = fmaf(r15, r16, r12 * r19);
    r21 = r13 * r18;
    r20 = fmaf(r5, r21, r20);
    r20 = fmaf(r14, r17, r20);
    r21 = 2.00000000000000000e+00;
    r22 = fmaf(r13, r16, r14 * r19);
    r23 = r12 * r17;
    r22 = fmaf(r5, r23, r22);
    r22 = fmaf(r15, r18, r22);
    r23 = r21 * r22;
    r24 = r20 * r23;
    r25 = r14 * r16;
    r25 = fmaf(r5, r25, r13 * r19);
    r25 = fmaf(r15, r17, r25);
    r25 = fmaf(r12, r18, r25);
    r26 = fmaf(r13, r17, r12 * r16);
    r26 = fmaf(r14, r18, r26);
    r15 = fmaf(r15, r19, r5 * r26);
    r26 = r25 * r15;
    r27 = fmaf(r21, r26, r24);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r28,
                       r29,
                       r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = -2.00000000000000000e+00;
    r32 = r18 * r18;
    r32 = r31 * r32;
    r33 = 1.00000000000000000e+00;
    r34 = r17 * r17;
    r34 = fmaf(r31, r34, r33);
    r35 = r32 + r34;
    r35 = fmaf(r28, r35, r11 * r27);
    r27 = r16 * r18;
    r27 = r27 * r21;
    r36 = r17 * r19;
    r36 = fmaf(r21, r36, r27);
    r37 = r16 * r17;
    r37 = r37 * r21;
    r38 = r19 * r31;
    r39 = fmaf(r18, r38, r37);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = 2.71828182845904523536;
    r40 = powf(r41, r40);
    r41 = r31 * r25;
    r41 = r41 * r25;
    r42 = r33 + r41;
    r43 = r31 * r22;
    r43 = r43 * r22;
    r42 = r42 + r43;
    r44 = r21 * r20;
    r44 = r44 * r25;
    r45 = r31 * r22;
    r45 = fmaf(r15, r45, r44);
    r35 = fmaf(r30, r36, r35);
    r35 = fmaf(r29, r39, r35);
    r35 = fmaf(r1, r40, r35);
    r35 = fmaf(r9, r42, r35);
    r35 = fmaf(r10, r45, r35);
    r45 = r8 * r35;
    r42 = 9.99999999999999955e-07;
    r26 = fmaf(r31, r26, r24);
    r24 = r16 * r16;
    r24 = r31 * r24;
    r34 = r24 + r34;
    r34 = fmaf(r30, r34, r9 * r26);
    r27 = fmaf(r17, r38, r27);
    r26 = r16 * r19;
    r1 = r17 * r18;
    r1 = r1 * r21;
    r26 = fmaf(r21, r26, r1);
    r39 = r21 * r20;
    r25 = r25 * r23;
    r39 = fmaf(r15, r39, r25);
    r41 = r33 + r41;
    r36 = r20 * r20;
    r36 = r31 * r36;
    r41 = r41 + r36;
    r34 = fmaf(r28, r27, r34);
    r34 = fmaf(r29, r26, r34);
    r34 = fmaf(r7, r40, r34);
    r34 = fmaf(r10, r39, r34);
    r34 = fmaf(r11, r41, r34);
    r41 = copysign(1.0, r34);
    r41 = fmaf(r42, r41, r34);
    r41 = 1.0 / r41;
    r3 = fmaf(r41, r45, r3);
    r3 = r3 * r3;
  };
  LoadUnique<1, float, float>(reprojection_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r45);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = r45 * r45;
    r45 = 1.0 / r45;
    r5 = fmaf(r4, r5, r2);
    r43 = r33 + r43;
    r43 = r43 + r36;
    r32 = r33 + r32;
    r32 = r32 + r24;
    r32 = fmaf(r29, r32, r10 * r43);
    r29 = r18 * r19;
    r29 = fmaf(r21, r29, r37);
    r38 = fmaf(r16, r38, r1);
    r1 = r31 * r20;
    r1 = fmaf(r15, r1, r25);
    r23 = fmaf(r15, r23, r44);
    r32 = fmaf(r28, r29, r32);
    r32 = fmaf(r30, r38, r32);
    r32 = fmaf(r6, r40, r32);
    r32 = fmaf(r11, r1, r32);
    r32 = fmaf(r9, r23, r32);
    r23 = r0 * r32;
    r5 = fmaf(r41, r23, r5);
    r5 = r5 * r5;
    r23 = r3 + r5;
    r45 = fmaf(r23, r45, r33);
    r45 = sqrtf(r45);
    r45 = r33 + r45;
    r45 = 1.0 / r45;
    r45 = r21 * r45;
    r45 = fmaf(r5, r45, r3 * r45);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r45);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void RowFixedRigPinholeScore(float* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             float* sensor_calibration,
                             unsigned int sensor_calibration_num_alloc,
                             SharedIndex* sensor_calibration_indices,
                             const float* const sensor_from_rig_log_scale,
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
  RowFixedRigPinholeScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_calibration,
      sensor_calibration_num_alloc,
      sensor_calibration_indices,
      sensor_from_rig_log_scale,
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