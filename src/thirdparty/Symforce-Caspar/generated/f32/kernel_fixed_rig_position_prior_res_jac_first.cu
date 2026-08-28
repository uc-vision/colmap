#include "kernel_fixed_rig_position_prior_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigPositionPriorResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* position,
        unsigned int position_num_alloc,
        float* sqrt_information,
        unsigned int sqrt_information_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;

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
  if (global_thread_idx < problem_size) {
    r13 = r11 * r11;
    r14 = r8 * r13;
    r15 = 1.00000000000000000e+00;
    r16 = r10 * r10;
    r17 = fmaf(r8, r16, r15);
    r18 = r14 + r17;
    r19 = r11 * r12;
    r20 = 2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = r9 * r20;
    r22 = r10 * r21;
    r23 = r19 + r22;
    r23 = r6 * r23;
    r18 = fmaf(r5, r18, r23);
    r24 = r11 * r21;
    r25 = r12 * r8;
    r26 = r10 * r25;
    r27 = r24 + r26;
    r18 = fmaf(r7, r27, r18);
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r28, r29, r30);
    r28 = fmaf(r28, r4, r4 * r18);
    ReadIdx4<1024, float, float, float4>(sqrt_information,
                                         4 * sqrt_information_num_alloc,
                                         global_thread_idx,
                                         r18,
                                         r31,
                                         r32,
                                         r33);
    r34 = r9 * r9;
    r34 = r34 * r8;
    r17 = r34 + r17;
    r35 = r10 * r11;
    r35 = r35 * r20;
    r36 = r9 * r25;
    r37 = r35 + r36;
    r17 = fmaf(r6, r37, r7 * r17);
    r38 = r10 * r12;
    r38 = r38 * r20;
    r24 = r38 + r24;
    r24 = r5 * r24;
    r17 = r17 + r24;
    r30 = fmaf(r30, r4, r4 * r17);
    r17 = fmaf(r32, r30, r0 * r28);
    r14 = r15 + r14;
    r14 = r14 + r34;
    r34 = r12 * r21;
    r35 = r35 + r34;
    r14 = fmaf(r7, r35, r6 * r14);
    r25 = r11 * r25;
    r22 = r22 + r25;
    r14 = fmaf(r5, r22, r14);
    r29 = fmaf(r29, r4, r4 * r14);
    r17 = fmaf(r3, r29, r17);
    r14 = fmaf(r33, r30, r1 * r28);
    r14 = fmaf(r18, r29, r14);
    ReadIdx1<1024, float, float, float>(sqrt_information,
                                        8 * sqrt_information_num_alloc,
                                        global_thread_idx,
                                        r15);
    r30 = fmaf(r15, r30, r2 * r28);
    r30 = fmaf(r31, r29, r30);
    WriteIdx3<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r17, r14, r30);
    r29 = fmaf(r14, r14, r30 * r30);
    r29 = fmaf(r17, r17, r29);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r29);
  if (global_thread_idx < problem_size) {
    r29 = r4 * r17;
    r28 = r6 * r4;
    r7 = r7 * r4;
    r39 = r4 * r16;
    r40 = r13 + r39;
    r41 = r12 * r12;
    r42 = r9 * r9;
    r42 = r42 * r4;
    r43 = r41 + r42;
    r44 = r40 + r43;
    r44 = fmaf(r44, r7, r37 * r28);
    r44 = fmaf(r4, r24, r44);
    r24 = r6 * r4;
    r28 = r9 * r9;
    r12 = r12 * r12;
    r12 = r12 * r4;
    r37 = r28 + r12;
    r40 = r40 + r37;
    r45 = r5 * r4;
    r46 = r9 * r10;
    r46 = r46 * r8;
    r19 = r19 + r46;
    r45 = fmaf(r19, r45, r40 * r24);
    r10 = r10 * r11;
    r10 = r10 * r8;
    r36 = r10 + r36;
    r45 = fmaf(r36, r7, r45);
    r24 = fmaf(r32, r45, r3 * r44);
    r40 = r4 * r30;
    r47 = fmaf(r15, r45, r31 * r44);
    r40 = fmaf(r47, r40, r24 * r29);
    r29 = r4 * r14;
    r45 = fmaf(r33, r45, r18 * r44);
    r40 = fmaf(r45, r29, r40);
    r29 = r4 * r14;
    r44 = r6 * r4;
    r34 = r10 + r34;
    r10 = r5 * r4;
    r11 = r9 * r11;
    r11 = r11 * r8;
    r26 = r11 + r26;
    r10 = fmaf(r26, r10, r34 * r44);
    r44 = r4 * r13;
    r8 = r16 + r44;
    r37 = r37 + r8;
    r10 = fmaf(r37, r7, r10);
    r37 = r5 * r4;
    r41 = r28 + r41;
    r41 = r41 + r39;
    r41 = r41 + r44;
    r23 = fmaf(r4, r23, r41 * r37);
    r23 = fmaf(r27, r7, r23);
    r27 = fmaf(r33, r23, r1 * r10);
    r37 = r4 * r30;
    r41 = fmaf(r15, r23, r2 * r10);
    r37 = fmaf(r41, r37, r27 * r29);
    r29 = r4 * r17;
    r23 = fmaf(r32, r23, r0 * r10);
    r37 = fmaf(r23, r29, r37);
    r29 = r4 * r30;
    r10 = r5 * r4;
    r44 = r16 + r13;
    r44 = r44 + r42;
    r44 = r44 + r12;
    r12 = r6 * r4;
    r25 = r46 + r25;
    r12 = fmaf(r25, r12, r44 * r10);
    r11 = r38 + r11;
    r12 = fmaf(r11, r7, r12);
    r38 = r6 * r4;
    r8 = r43 + r8;
    r43 = r5 * r4;
    r43 = fmaf(r22, r43, r8 * r38);
    r43 = fmaf(r35, r7, r43);
    r7 = fmaf(r2, r43, r31 * r12);
    r35 = r4 * r17;
    r38 = fmaf(r0, r43, r3 * r12);
    r35 = fmaf(r38, r35, r7 * r29);
    r29 = r4 * r14;
    r43 = fmaf(r1, r43, r18 * r12);
    r35 = fmaf(r43, r29, r35);
    r29 = r4 * r17;
    r13 = r20 * r13;
    r16 = fmaf(r20, r16, r4);
    r20 = r13 + r16;
    r12 = fmaf(r32, r26, r0 * r20);
    r12 = fmaf(r3, r19, r12);
    r22 = r4 * r30;
    r8 = fmaf(r15, r26, r2 * r20);
    r8 = fmaf(r31, r19, r8);
    r22 = fmaf(r8, r22, r12 * r29);
    r29 = r4 * r14;
    r26 = fmaf(r33, r26, r1 * r20);
    r26 = fmaf(r18, r19, r26);
    r22 = fmaf(r26, r29, r22);
    WriteSum4<float, float>((float*)inout_shared, r40, r37, r35, r22);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r4 * r14;
    r13 = r4 + r13;
    r21 = r9 * r21;
    r13 = r13 + r21;
    r9 = fmaf(r1, r25, r18 * r13);
    r9 = fmaf(r33, r34, r9);
    r35 = r4 * r17;
    r37 = fmaf(r0, r25, r3 * r13);
    r37 = fmaf(r32, r34, r37);
    r35 = fmaf(r37, r35, r9 * r22);
    r22 = r4 * r30;
    r25 = fmaf(r2, r25, r31 * r13);
    r25 = fmaf(r15, r34, r25);
    r35 = fmaf(r25, r22, r35);
    r22 = r4 * r30;
    r16 = r21 + r16;
    r2 = fmaf(r2, r11, r15 * r16);
    r2 = fmaf(r31, r36, r2);
    r31 = r4 * r14;
    r1 = fmaf(r1, r11, r33 * r16);
    r1 = fmaf(r18, r36, r1);
    r31 = fmaf(r1, r31, r2 * r22);
    r22 = r4 * r17;
    r11 = fmaf(r0, r11, r32 * r16);
    r11 = fmaf(r3, r36, r11);
    r31 = fmaf(r11, r22, r31);
    WriteSum2<float, float>((float*)inout_shared, r35, r31);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r45, r45, r47 * r47);
    r31 = fmaf(r24, r24, r31);
    r35 = fmaf(r23, r23, r41 * r41);
    r35 = fmaf(r27, r27, r35);
    r22 = fmaf(r7, r7, r38 * r38);
    r22 = fmaf(r43, r43, r22);
    r36 = fmaf(r8, r8, r26 * r26);
    r36 = fmaf(r12, r12, r36);
    WriteSum4<float, float>((float*)inout_shared, r31, r35, r22, r36);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r25, r25, r9 * r9);
    r36 = fmaf(r37, r37, r36);
    r22 = fmaf(r2, r2, r11 * r11);
    r22 = fmaf(r1, r1, r22);
    WriteSum2<float, float>((float*)inout_shared, r36, r22);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r47, r41, r24 * r23);
    r22 = fmaf(r45, r27, r22);
    r36 = fmaf(r45, r43, r47 * r7);
    r36 = fmaf(r24, r38, r36);
    r35 = fmaf(r47, r8, r45 * r26);
    r35 = fmaf(r24, r12, r35);
    r31 = fmaf(r47, r25, r45 * r9);
    r31 = fmaf(r24, r37, r31);
    WriteSum4<float, float>((float*)inout_shared, r22, r36, r35, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r47, r2, r24 * r11);
    r47 = fmaf(r45, r1, r47);
    r45 = fmaf(r41, r7, r23 * r38);
    r45 = fmaf(r27, r43, r45);
    r24 = fmaf(r41, r8, r27 * r26);
    r24 = fmaf(r23, r12, r24);
    r31 = fmaf(r23, r37, r27 * r9);
    r31 = fmaf(r41, r25, r31);
    WriteSum4<float, float>((float*)inout_shared, r47, r45, r24, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r41, r2, r23 * r11);
    r41 = fmaf(r27, r1, r41);
    r27 = fmaf(r38, r12, r7 * r8);
    r27 = fmaf(r43, r26, r27);
    r23 = fmaf(r7, r25, r43 * r9);
    r23 = fmaf(r38, r37, r23);
    r7 = fmaf(r7, r2, r43 * r1);
    r7 = fmaf(r38, r11, r7);
    WriteSum4<float, float>((float*)inout_shared, r41, r27, r23, r7);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r8, r25, r12 * r37);
    r7 = fmaf(r26, r9, r7);
    r8 = fmaf(r8, r2, r26 * r1);
    r8 = fmaf(r12, r11, r8);
    r2 = fmaf(r25, r2, r37 * r11);
    r2 = fmaf(r9, r1, r2);
    WriteSum3<float, float>((float*)inout_shared, r7, r8, r2);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void FixedRigPositionPriorResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* position,
    unsigned int position_num_alloc,
    float* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigPositionPriorResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      position,
      position_num_alloc,
      sqrt_information,
      sqrt_information_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar