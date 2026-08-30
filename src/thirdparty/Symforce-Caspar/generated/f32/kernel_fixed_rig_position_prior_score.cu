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
    FixedRigPositionPriorScoreKernel(float* pose,
                                     unsigned int pose_num_alloc,
                                     SharedIndex* pose_indices,
                                     float* position,
                                     unsigned int position_num_alloc,
                                     float* sqrt_information,
                                     unsigned int sqrt_information_num_alloc,
                                     const float* const position_loss_scale,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28;

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
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r10;
    r13 = -2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = 1.00000000000000000e+00;
    r15 = r9 * r9;
    r15 = fmaf(r13, r15, r14);
    r16 = r12 + r15;
    r17 = r8 * r9;
    r18 = 2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = r10 * r18;
    r20 = fmaf(r11, r19, r17);
    r20 = fmaf(r6, r20, r5 * r16);
    r16 = r8 * r19;
    r21 = r11 * r13;
    r22 = fmaf(r9, r21, r16);
    r20 = fmaf(r7, r22, r20);
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r22, r23, r24);
    r22 = fmaf(r22, r4, r4 * r20);
    ReadIdx4<1024, float, float, float4>(sqrt_information,
                                         4 * sqrt_information_num_alloc,
                                         global_thread_idx,
                                         r20,
                                         r25,
                                         r26,
                                         r27);
    r28 = r8 * r8;
    r28 = r13 * r28;
    r15 = r28 + r15;
    r19 = r9 * r19;
    r13 = fmaf(r8, r21, r19);
    r13 = fmaf(r6, r13, r7 * r15);
    r15 = r9 * r11;
    r15 = fmaf(r18, r15, r16);
    r13 = fmaf(r5, r15, r13);
    r24 = fmaf(r24, r4, r4 * r13);
    r26 = fmaf(r26, r24, r0 * r22);
    r12 = r14 + r12;
    r12 = r12 + r28;
    r28 = r8 * r11;
    r28 = fmaf(r18, r28, r19);
    r28 = fmaf(r7, r28, r6 * r12);
    r21 = fmaf(r10, r21, r17);
    r28 = fmaf(r5, r21, r28);
    r4 = fmaf(r23, r4, r4 * r28);
    r26 = fmaf(r3, r4, r26);
    r26 = r26 * r26;
  };
  LoadUnique<1, float, float>(position_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = r3 * r3;
    r3 = 1.0 / r3;
    ReadIdx1<1024, float, float, float>(sqrt_information,
                                        8 * sqrt_information_num_alloc,
                                        global_thread_idx,
                                        r23);
    r23 = fmaf(r23, r24, r2 * r22);
    r23 = fmaf(r25, r4, r23);
    r23 = r23 * r23;
    r25 = r26 + r23;
    r24 = fmaf(r27, r24, r1 * r22);
    r24 = fmaf(r20, r4, r24);
    r24 = r24 * r24;
    r25 = r25 + r24;
    r3 = fmaf(r25, r3, r14);
    r3 = sqrtf(r3);
    r3 = r14 + r3;
    r3 = 1.0 / r3;
    r3 = r18 * r3;
    r23 = fmaf(r23, r3, r26 * r3);
    r23 = fmaf(r24, r3, r23);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r23);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void FixedRigPositionPriorScore(float* pose,
                                unsigned int pose_num_alloc,
                                SharedIndex* pose_indices,
                                float* position,
                                unsigned int position_num_alloc,
                                float* sqrt_information,
                                unsigned int sqrt_information_num_alloc,
                                const float* const position_loss_scale,
                                float* const out_rTr,
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
      position_loss_scale,
      out_rTr,
      problem_size);
}

}  // namespace caspar