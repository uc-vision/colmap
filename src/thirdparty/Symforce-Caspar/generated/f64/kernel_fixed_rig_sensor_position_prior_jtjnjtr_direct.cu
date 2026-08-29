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
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        const double* const sensor_from_rig_log_scale_njtr,
        double* sensor_from_rig_log_scale_jac,
        unsigned int sensor_from_rig_log_scale_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_sensor_from_rig_log_scale_njtr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_sensor_from_rig_log_scale_njtr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale_njtr, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = r0 * r2;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig_log_scale_jac,
        2 * sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r4);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r5, r6);
    r7 = r6 * r2;
    ReadIdx2<1024, double, double, double2>(
        sensor_from_rig_log_scale_jac,
        0 * sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r8,
        r9);
    r7 = fma(r9, r7, r4 * r3);
    r3 = r2 * r8;
    r7 = fma(r5, r3, r7);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r10, r11);
    r12 = r10 * r2;
    r13 = r11 * r2;
    r13 = fma(r4, r13, r9 * r12);
    r13 = fma(r1, r3, r13);
    WriteSum2<double, double>((double*)inout_shared, r7, r13);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r13, r7);
    r12 = r13 * r2;
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r14, r15);
    r16 = r15 * r2;
    r16 = fma(r9, r16, r4 * r12);
    r16 = fma(r14, r3, r16);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r12, r17);
    r18 = r12 * r2;
    r19 = r17 * r2;
    r19 = fma(r4, r19, r9 * r18);
    r19 = fma(r7, r3, r19);
    WriteSum2<double, double>((double*)inout_shared, r16, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 12 * pose_jac_num_alloc, global_thread_idx, r19, r16);
    r18 = r16 * r2;
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 14 * pose_jac_num_alloc, global_thread_idx, r20, r21);
    r22 = r20 * r2;
    r22 = fma(r4, r22, r9 * r18);
    r22 = fma(r19, r3, r22);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 16 * pose_jac_num_alloc, global_thread_idx, r18, r23);
    r24 = r18 * r2;
    r25 = r23 * r2;
    r25 = fma(r4, r25, r9 * r24);
    r25 = fma(r21, r3, r25);
    WriteSum2<double, double>((double*)inout_shared, r22, r25);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(pose_njtr,
                                4 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r25,
                        r22);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r3,
                        r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r26 = fma(r24, r17, r22 * r23);
  };
  LoadShared<2, double, double>(pose_njtr,
                                0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r27,
                        r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r26 = fma(r25, r20, r26);
    r26 = fma(r3, r13, r26);
    r26 = fma(r28, r11, r26);
    r26 = fma(r27, r0, r26);
    r29 = fma(r24, r12, r22 * r18);
    r29 = fma(r25, r16, r29);
    r29 = fma(r3, r15, r29);
    r29 = fma(r28, r10, r29);
    r29 = fma(r27, r6, r29);
    r29 = fma(r9, r29, r4 * r26);
    r19 = fma(r25, r19, r22 * r21);
    r19 = fma(r3, r14, r19);
    r19 = fma(r27, r5, r19);
    r19 = fma(r28, r1, r19);
    r19 = fma(r24, r7, r19);
    r29 = fma(r8, r19, r29);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r29);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_njtr_local,
                        out_sensor_from_rig_log_scale_njtr,
                        1);
}

void FixedRigSensorPositionPriorJtjnjtrDirect(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    const double* const sensor_from_rig_log_scale_njtr,
    double* sensor_from_rig_log_scale_jac,
    unsigned int sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
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