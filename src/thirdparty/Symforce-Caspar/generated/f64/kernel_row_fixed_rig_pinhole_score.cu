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
    RowFixedRigPinholeScoreKernel(double* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  double* sensor_calibration,
                                  unsigned int sensor_calibration_num_alloc,
                                  SharedIndex* sensor_calibration_indices,
                                  const double* const sensor_from_rig_log_scale,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;
  LoadShared<2, double, double>(sensor_calibration,
                                8 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
  };
  LoadShared<2, double, double>(sensor_calibration,
                                6 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r1,
                        r5);
  };
  __syncthreads();
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r7, r8);
  };
  __syncthreads();
  LoadShared<2, double, double>(sensor_calibration,
                                2 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r11, r12);
  };
  __syncthreads();
  LoadShared<2, double, double>(sensor_calibration,
                                0 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r13,
                        r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fma(r12, r13, r7 * r10);
    r16 = r8 * r9;
    r15 = fma(r4, r16, r15);
    r15 = fma(r11, r14, r15);
    r16 = 2.00000000000000000e+00;
    r17 = fma(r8, r13, r11 * r10);
    r18 = r7 * r14;
    r17 = fma(r4, r18, r17);
    r17 = fma(r12, r9, r17);
    r18 = r16 * r17;
    r19 = r15 * r18;
    r20 = r11 * r13;
    r20 = fma(r4, r20, r8 * r10);
    r20 = fma(r12, r14, r20);
    r20 = fma(r7, r9, r20);
    r21 = fma(r8, r14, r7 * r13);
    r21 = fma(r11, r9, r21);
    r12 = fma(r12, r10, r4 * r21);
    r21 = r20 * r12;
    r22 = fma(r16, r21, r19);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r23, r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r25 = -2.00000000000000000e+00;
    r26 = r9 * r9;
    r26 = r25 * r26;
    r27 = 1.00000000000000000e+00;
    r28 = r14 * r14;
    r28 = fma(r25, r28, r27);
    r29 = r26 + r28;
    r29 = fma(r23, r29, r6 * r22);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r14 * r10;
    r31 = r13 * r9;
    r31 = r31 * r16;
    r30 = fma(r16, r30, r31);
    r32 = r13 * r14;
    r32 = r32 * r16;
    r33 = r10 * r25;
    r34 = fma(r9, r33, r32);
  };
  LoadShared<2, double, double>(sensor_calibration,
                                4 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r35,
                        r36);
  };
  __syncthreads();
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = 2.71828182845904523536;
    r37 = pow(r38, r37);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r38, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r20 * r20;
    r40 = r40 * r25;
    r41 = r27 + r40;
    r42 = r17 * r17;
    r42 = r42 * r25;
    r41 = r41 + r42;
    r43 = r16 * r15;
    r43 = r43 * r20;
    r44 = r17 * r12;
    r44 = fma(r25, r44, r43);
    r29 = fma(r22, r30, r29);
    r29 = fma(r24, r34, r29);
    r29 = fma(r35, r37, r29);
    r29 = fma(r38, r41, r29);
    r29 = fma(r39, r44, r29);
    r44 = r5 * r29;
    r41 = 1.00000000000000008e-15;
    r21 = fma(r25, r21, r19);
    r19 = r13 * r13;
    r19 = r25 * r19;
    r28 = r19 + r28;
    r28 = fma(r22, r28, r38 * r21);
    r31 = fma(r14, r33, r31);
    r21 = r13 * r10;
    r35 = r14 * r9;
    r35 = r35 * r16;
    r21 = fma(r16, r21, r35);
    r34 = r16 * r15;
    r20 = r20 * r18;
    r34 = fma(r12, r34, r20);
    r40 = r27 + r40;
    r30 = r15 * r15;
    r30 = r25 * r30;
    r40 = r40 + r30;
    r28 = fma(r23, r31, r28);
    r28 = fma(r24, r21, r28);
    r28 = fma(r1, r37, r28);
    r28 = fma(r39, r34, r28);
    r28 = fma(r6, r40, r28);
    r40 = copysign(1.0, r28);
    r40 = fma(r41, r40, r28);
    r40 = 1.0 / r40;
    r2 = fma(r40, r44, r2);
  };
  LoadShared<1, double, double>(sensor_calibration,
                                10 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r44);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = fma(r3, r4, r44);
    r42 = r27 + r42;
    r42 = r42 + r30;
    r26 = r27 + r26;
    r26 = r26 + r19;
    r26 = fma(r24, r26, r39 * r42);
    r24 = r9 * r10;
    r24 = fma(r16, r24, r32);
    r33 = fma(r13, r33, r35);
    r35 = r15 * r12;
    r35 = fma(r25, r35, r20);
    r18 = fma(r12, r18, r43);
    r26 = fma(r23, r24, r26);
    r26 = fma(r22, r33, r26);
    r26 = fma(r36, r37, r26);
    r26 = fma(r6, r35, r26);
    r26 = fma(r38, r18, r26);
    r18 = r0 * r26;
    r4 = fma(r40, r18, r4);
    r4 = fma(r4, r4, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void RowFixedRigPinholeScore(double* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             double* sensor_calibration,
                             unsigned int sensor_calibration_num_alloc,
                             SharedIndex* sensor_calibration_indices,
                             const double* const sensor_from_rig_log_scale,
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