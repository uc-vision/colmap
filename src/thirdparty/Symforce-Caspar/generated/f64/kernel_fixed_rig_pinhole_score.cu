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
    FixedRigPinholeScoreKernel(double* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               double* sensor_from_rig,
                               unsigned int sensor_from_rig_num_alloc,
                               const double* const sensor_from_rig_log_scale,
                               double* calib,
                               unsigned int calib_num_alloc,
                               double* point,
                               unsigned int point_num_alloc,
                               SharedIndex* point_indices,
                               double* pixel,
                               unsigned int pixel_num_alloc,
                               double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib, 2 * calib_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r5);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    r17 = r13 * r15;
    r17 = fma(r4, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r18 = r8 * r17;
    r18 = r18 * r17;
    r19 = 1.00000000000000000e+00;
    r20 = fma(r10, r15, r13 * r12);
    r21 = r9 * r16;
    r20 = fma(r4, r21, r20);
    r20 = fma(r14, r11, r20);
    r21 = r20 * r20;
    r21 = fma(r8, r21, r19);
    r22 = r18 + r21;
    r23 = fma(r14, r15, r9 * r12);
    r24 = r10 * r11;
    r23 = fma(r4, r24, r23);
    r23 = fma(r13, r16, r23);
    r24 = 2.00000000000000000e+00;
    r25 = r24 * r17;
    r26 = r23 * r25;
    r27 = fma(r10, r16, r9 * r15);
    r27 = fma(r13, r11, r27);
    r27 = fma(r4, r27, r14 * r12);
    r14 = r8 * r27;
    r28 = fma(r20, r14, r26);
    r28 = fma(r7, r28, r6 * r22);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r20 * r24;
    r29 = r29 * r23;
    r30 = fma(r27, r25, r29);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r15 * r11;
    r32 = r32 * r24;
    r33 = r16 * r12;
    r34 = fma(r24, r33, r32);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r35, r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r11 * r12;
    r38 = r15 * r16;
    r38 = r38 * r24;
    r37 = fma(r8, r37, r38);
    r39 = r11 * r11;
    r39 = r8 * r39;
    r40 = r19 + r39;
    r41 = r16 * r16;
    r41 = r41 * r8;
    r40 = r40 + r41;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r42,
                                            r43);
  };
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r44);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = 2.71828182845904523536;
    r44 = pow(r45, r44);
    r28 = fma(r22, r30, r28);
    r28 = fma(r31, r34, r28);
    r28 = fma(r36, r37, r28);
    r28 = fma(r35, r40, r28);
    r28 = fma(r42, r44, r28);
    r42 = r0 * r28;
    r40 = 1.00000000000000008e-15;
    r17 = fma(r17, r14, r29);
    r33 = fma(r8, r33, r32);
    r33 = fma(r35, r33, r6 * r17);
    r41 = r19 + r41;
    r17 = r15 * r15;
    r17 = r8 * r17;
    r41 = r41 + r17;
    r32 = r16 * r11;
    r32 = r32 * r24;
    r29 = r15 * r12;
    r29 = fma(r24, r29, r32);
    r37 = r24 * r23;
    r25 = r20 * r25;
    r37 = fma(r27, r37, r25);
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r34);
    r18 = r19 + r18;
    r30 = r23 * r23;
    r30 = r8 * r30;
    r18 = r18 + r30;
    r33 = fma(r31, r41, r33);
    r33 = fma(r36, r29, r33);
    r33 = fma(r7, r37, r33);
    r33 = fma(r34, r44, r33);
    r33 = fma(r22, r18, r33);
    r18 = copysign(1.0, r33);
    r18 = fma(r40, r18, r33);
    r18 = 1.0 / r18;
    r2 = fma(r18, r42, r2);
    r4 = fma(r3, r4, r1);
    r3 = r20 * r24;
    r3 = fma(r27, r3, r26);
    r26 = r11 * r12;
    r26 = fma(r24, r26, r38);
    r26 = fma(r35, r26, r6 * r3);
    r39 = r19 + r39;
    r39 = r39 + r17;
    r17 = r15 * r12;
    r17 = fma(r8, r17, r32);
    r14 = fma(r23, r14, r25);
    r21 = r30 + r21;
    r26 = fma(r36, r39, r26);
    r26 = fma(r31, r17, r26);
    r26 = fma(r22, r14, r26);
    r26 = fma(r43, r44, r26);
    r26 = fma(r7, r21, r26);
    r21 = r5 * r26;
    r4 = fma(r18, r21, r4);
    r4 = fma(r4, r4, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedRigPinholeScore(double* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          double* sensor_from_rig,
                          unsigned int sensor_from_rig_num_alloc,
                          const double* const sensor_from_rig_log_scale,
                          double* calib,
                          unsigned int calib_num_alloc,
                          double* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          double* pixel,
                          unsigned int pixel_num_alloc,
                          double* const out_rTr,
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
                                                 out_rTr,
                                                 problem_size);
}

}  // namespace caspar