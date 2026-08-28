#include "kernel_fixed_rig_position_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigPositionPriorScoreKernel(double* pose,
                                     unsigned int pose_num_alloc,
                                     SharedIndex* pose_indices,
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
      r16, r17, r18, r19, r20, r21, r22;

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
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = r5 * r5;
    r8 = -2.00000000000000000e+00;
    r7 = r7 * r8;
    r9 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r11 * r11;
    r12 = fma(r8, r12, r9);
    r13 = r7 + r12;
    r14 = r10 * r11;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
    r16 = r5 * r15;
    r17 = fma(r6, r16, r14);
    r17 = fma(r4, r17, r3 * r13);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r10 * r16;
    r19 = r6 * r8;
    r20 = fma(r11, r19, r18);
    r17 = fma(r13, r20, r17);
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r20, r21);
    r20 = fma(r20, r2, r2 * r17);
    ReadIdx1<1024, double, double, double>(sqrt_information,
                                           8 * sqrt_information_num_alloc,
                                           global_thread_idx,
                                           r17);
    r22 = r10 * r10;
    r22 = r8 * r22;
    r12 = r22 + r12;
    r16 = r11 * r16;
    r8 = fma(r10, r19, r16);
    r8 = fma(r4, r8, r13 * r12);
    r12 = r11 * r6;
    r12 = fma(r15, r12, r18);
    r8 = fma(r3, r12, r8);
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r12);
    r12 = fma(r12, r2, r2 * r8);
    r17 = fma(r17, r12, r0 * r20);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            4 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r8);
    r7 = r9 + r7;
    r7 = r7 + r22;
    r22 = r10 * r6;
    r22 = fma(r15, r22, r16);
    r22 = fma(r13, r22, r4 * r7);
    r19 = fma(r5, r19, r14);
    r22 = fma(r3, r19, r22);
    r2 = fma(r21, r2, r2 * r22);
    r17 = fma(r8, r2, r17);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            0 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r8,
                                            r21);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            6 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r22,
                                            r19);
    r19 = fma(r19, r12, r21 * r20);
    r19 = fma(r0, r2, r19);
    r19 = fma(r19, r19, r17 * r17);
    r12 = fma(r22, r12, r8 * r20);
    r12 = fma(r1, r2, r12);
    r19 = fma(r12, r12, r19);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r19);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedRigPositionPriorScore(double* pose,
                                unsigned int pose_num_alloc,
                                SharedIndex* pose_indices,
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
  FixedRigPositionPriorScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      position,
      position_num_alloc,
      sqrt_information,
      sqrt_information_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar