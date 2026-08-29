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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
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
    r4 = fmaf(r4, r5, r2);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r2,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -2.00000000000000000e+00;
  };
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
  LoadShared<4, float, float>(sensor_calibration,
                              0 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r10, r13, r11 * r16);
    r18 = r9 * r14;
    r17 = fmaf(r5, r18, r17);
    r17 = fmaf(r12, r15, r17);
    r18 = r17 * r17;
    r18 = r8 * r18;
    r19 = 1.00000000000000000e+00;
    r20 = fmaf(r12, r13, r9 * r16);
    r21 = r10 * r15;
    r20 = fmaf(r5, r21, r20);
    r20 = fmaf(r11, r14, r20);
    r21 = r20 * r20;
    r21 = fmaf(r8, r21, r19);
    r22 = r18 + r21;
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r23,
                       r24,
                       r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r26 = r13 * r13;
    r26 = r8 * r26;
    r27 = r19 + r26;
    r28 = r15 * r15;
    r28 = r28 * r8;
    r27 = r27 + r28;
    r27 = fmaf(r24, r27, r6 * r22);
    r22 = r13 * r14;
    r29 = 2.00000000000000000e+00;
    r22 = r22 * r29;
    r30 = r15 * r29;
    r31 = fmaf(r16, r30, r22);
    r32 = r14 * r30;
    r33 = r16 * r8;
    r34 = fmaf(r13, r33, r32);
  };
  LoadShared<4, float, float>(sensor_calibration,
                              4 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r35,
                       r36,
                       r37,
                       r38);
  };
  __syncthreads();
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = 2.71828182845904523536;
    r39 = powf(r40, r39);
    r40 = r29 * r17;
    r41 = r11 * r13;
    r41 = fmaf(r5, r41, r10 * r16);
    r41 = fmaf(r12, r14, r41);
    r41 = fmaf(r9, r15, r41);
    r40 = r40 * r41;
    r42 = r20 * r8;
    r43 = fmaf(r10, r14, r9 * r13);
    r43 = fmaf(r11, r15, r43);
    r12 = fmaf(r12, r16, r5 * r43);
    r42 = fmaf(r12, r42, r40);
    r43 = r29 * r20;
    r43 = r43 * r41;
    r44 = r29 * r17;
    r44 = fmaf(r12, r44, r43);
    r27 = fmaf(r23, r31, r27);
    r27 = fmaf(r25, r34, r27);
    r27 = fmaf(r36, r39, r27);
    r27 = fmaf(r7, r42, r27);
    r27 = fmaf(r2, r44, r27);
    r44 = r0 * r27;
    r42 = 9.99999999999999955e-07;
    r36 = r29 * r17;
    r36 = r36 * r20;
    r34 = r41 * r12;
    r31 = fmaf(r8, r34, r36);
    r26 = r19 + r26;
    r45 = r14 * r14;
    r45 = r8 * r45;
    r26 = r26 + r45;
    r26 = fmaf(r25, r26, r2 * r31);
    r30 = r13 * r30;
    r31 = fmaf(r14, r33, r30);
    r46 = r13 * r16;
    r46 = fmaf(r29, r46, r32);
    r32 = r29 * r20;
    r32 = fmaf(r12, r32, r40);
    r40 = r8 * r41;
    r40 = r40 * r41;
    r21 = r40 + r21;
    r26 = fmaf(r23, r31, r26);
    r26 = fmaf(r24, r46, r26);
    r26 = fmaf(r37, r39, r26);
    r26 = fmaf(r6, r32, r26);
    r26 = fmaf(r7, r21, r26);
    r21 = copysign(1.0, r26);
    r21 = fmaf(r42, r21, r26);
    r21 = 1.0 / r21;
    r4 = fmaf(r21, r44, r4);
    r5 = fmaf(r3, r5, r1);
    r34 = fmaf(r29, r34, r36);
    r28 = r19 + r28;
    r28 = r28 + r45;
    r28 = fmaf(r23, r28, r7 * r34);
    r23 = r14 * r16;
    r23 = fmaf(r29, r23, r30);
    r33 = fmaf(r15, r33, r22);
    r18 = r19 + r18;
    r18 = r18 + r40;
    r40 = r17 * r8;
    r40 = fmaf(r12, r40, r43);
    r28 = fmaf(r25, r23, r28);
    r28 = fmaf(r24, r33, r28);
    r28 = fmaf(r35, r39, r28);
    r28 = fmaf(r2, r18, r28);
    r28 = fmaf(r6, r40, r28);
    r40 = r38 * r28;
    r5 = fmaf(r21, r40, r5);
    r5 = fmaf(r5, r5, r4 * r4);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r5);
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
      out_rTr,
      problem_size);
}

}  // namespace caspar