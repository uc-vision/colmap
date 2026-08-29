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
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        const double* const sensor_from_rig_log_scale,
        double* position,
        unsigned int position_num_alloc,
        double* sqrt_information,
        unsigned int sqrt_information_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            2 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    r2 = -1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r3, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r5,
                                            r6);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r7,
                                            r8);
    r9 = r5 * r7;
    r10 = 2.00000000000000000e+00;
    r9 = r9 * r10;
    r11 = -2.00000000000000000e+00;
    r12 = r6 * r8;
    r13 = fma(r11, r12, r9);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = r6 * r6;
    r15 = r15 * r11;
    r16 = 1.00000000000000000e+00;
    r17 = r5 * r5;
    r17 = fma(r11, r17, r16);
    r18 = r15 + r17;
    r18 = fma(r14, r18, r3 * r13);
    r13 = r6 * r7;
    r13 = r13 * r10;
    r19 = r5 * r8;
    r19 = fma(r10, r19, r13);
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r20);
  };
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r21);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r22 = 2.71828182845904523536;
    r21 = pow(r22, r21);
    r18 = fma(r4, r19, r18);
    r18 = fma(r20, r21, r18);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r20, r19);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r22, r23);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r24 = fma(r23, r5, r20 * r8);
    r25 = r19 * r7;
    r24 = fma(r2, r25, r24);
    r24 = fma(r22, r6, r24);
    r25 = fma(r19, r5, r22 * r8);
    r26 = r20 * r6;
    r25 = fma(r2, r26, r25);
    r25 = fma(r23, r7, r25);
    r26 = r10 * r25;
    r27 = r24 * r26;
    r28 = r22 * r5;
    r28 = fma(r2, r28, r19 * r8);
    r28 = fma(r23, r6, r28);
    r28 = fma(r20, r7, r28);
    r29 = fma(r19, r6, r20 * r5);
    r29 = fma(r22, r7, r29);
    r29 = fma(r2, r29, r23 * r8);
    r23 = r11 * r29;
    r30 = fma(r28, r23, r27);
    r12 = fma(r10, r12, r9);
    r9 = r7 * r8;
    r31 = r5 * r6;
    r31 = r31 * r10;
    r9 = fma(r11, r9, r31);
    r9 = fma(r4, r9, r14 * r12);
    r15 = r16 + r15;
    r12 = r7 * r7;
    r12 = r11 * r12;
    r15 = r15 + r12;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r32,
                                            r33);
    r9 = fma(r3, r15, r9);
    r9 = fma(r32, r21, r9);
    r32 = r25 * r25;
    r32 = r32 * r11;
    r15 = r16 + r32;
    r34 = r28 * r28;
    r34 = r11 * r34;
    r15 = r15 + r34;
    r15 = fma(r15, r9, r30 * r18);
    r30 = r7 * r8;
    r30 = fma(r10, r30, r31);
    r17 = r12 + r17;
    r17 = fma(r4, r17, r3 * r30);
    r4 = r5 * r8;
    r4 = fma(r11, r4, r13);
    r17 = fma(r14, r4, r17);
    r17 = fma(r33, r21, r17);
    r21 = r10 * r24;
    r21 = r21 * r28;
    r33 = fma(r29, r26, r21);
    r15 = fma(r33, r17, r15);
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r33, r4);
    r33 = fma(r33, r2, r2 * r15);
    ReadIdx1<1024, double, double, double>(sqrt_information,
                                           8 * sqrt_information_num_alloc,
                                           global_thread_idx,
                                           r15);
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r14);
    r13 = r10 * r28;
    r13 = fma(r29, r13, r27);
    r26 = r28 * r26;
    r27 = fma(r24, r23, r26);
    r27 = fma(r17, r27, r9 * r13);
    r34 = r16 + r34;
    r13 = r24 * r24;
    r13 = r11 * r13;
    r34 = r34 + r13;
    r27 = fma(r18, r34, r27);
    r27 = fma(r2, r27, r14 * r2);
    r15 = fma(r15, r27, r0 * r33);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            4 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r14);
    r34 = r10 * r24;
    r34 = fma(r29, r34, r26);
    r23 = fma(r25, r23, r21);
    r23 = fma(r9, r23, r18 * r34);
    r32 = r16 + r32;
    r32 = r32 + r13;
    r23 = fma(r17, r32, r23);
    r2 = fma(r4, r2, r2 * r23);
    r15 = fma(r14, r2, r15);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            0 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r4);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            6 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r23,
                                            r32);
    r23 = fma(r23, r27, r14 * r33);
    r23 = fma(r1, r2, r23);
    r23 = fma(r23, r23, r15 * r15);
    r27 = fma(r32, r27, r4 * r33);
    r27 = fma(r0, r2, r27);
    r23 = fma(r27, r27, r23);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r23);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedRigSensorPositionPriorScore(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const double* const sensor_from_rig_log_scale,
    double* position,
    unsigned int position_num_alloc,
    double* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    double* const out_rTr,
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