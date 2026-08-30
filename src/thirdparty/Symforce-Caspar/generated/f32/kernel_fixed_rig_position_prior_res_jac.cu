#include "kernel_fixed_rig_position_prior_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedRigPositionPriorResJacKernel(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* position,
    unsigned int position_num_alloc,
    float* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    const float* const position_loss_scale,
    float* out_res,
    unsigned int out_res_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;

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
    r21 = r9 * r10;
    r21 = r21 * r20;
    r22 = r19 + r21;
    r22 = r6 * r22;
    r18 = fmaf(r5, r18, r22);
    r23 = r9 * r11;
    r23 = r23 * r20;
    r24 = r12 * r8;
    r25 = r10 * r24;
    r26 = r23 + r25;
    r26 = r7 * r26;
    r18 = r18 + r26;
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r27, r28, r29);
    r27 = fmaf(r27, r4, r4 * r18);
    ReadIdx4<1024, float, float, float4>(sqrt_information,
                                         4 * sqrt_information_num_alloc,
                                         global_thread_idx,
                                         r18,
                                         r30,
                                         r31,
                                         r32);
    r33 = r9 * r9;
    r34 = r8 * r33;
    r17 = r34 + r17;
    r35 = r10 * r11;
    r35 = r35 * r20;
    r36 = r9 * r24;
    r37 = r35 + r36;
    r17 = fmaf(r6, r37, r7 * r17);
    r38 = r10 * r12;
    r38 = r38 * r20;
    r23 = r23 + r38;
    r23 = r5 * r23;
    r17 = r17 + r23;
    r29 = fmaf(r29, r4, r4 * r17);
    r17 = fmaf(r31, r29, r0 * r27);
    r14 = r15 + r14;
    r14 = r14 + r34;
    r34 = r9 * r12;
    r34 = r34 * r20;
    r35 = r35 + r34;
    r14 = fmaf(r7, r35, r6 * r14);
    r24 = r11 * r24;
    r21 = r21 + r24;
    r14 = fmaf(r5, r21, r14);
    r28 = fmaf(r28, r4, r4 * r14);
    r17 = fmaf(r3, r28, r17);
  };
  LoadUnique<1, float, float>(position_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r14 * r14;
    r14 = 1.0 / r14;
    ReadIdx1<1024, float, float, float>(sqrt_information,
                                        8 * sqrt_information_num_alloc,
                                        global_thread_idx,
                                        r39);
    r40 = fmaf(r39, r29, r2 * r27);
    r40 = fmaf(r30, r28, r40);
    r29 = fmaf(r32, r29, r1 * r27);
    r29 = fmaf(r18, r28, r29);
    r28 = fmaf(r29, r29, r40 * r40);
    r28 = fmaf(r17, r17, r28);
    r28 = fmaf(r28, r14, r15);
    r27 = sqrtf(r28);
    r27 = r15 + r27;
    r15 = 1.0 / r27;
    r15 = r20 * r15;
    r41 = sqrtf(r15);
    r42 = r17 * r41;
    r43 = r29 * r41;
    r44 = r40 * r41;
    WriteIdx3<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r42, r43, r44);
    r44 = r7 * r4;
    r43 = r4 * r16;
    r42 = r13 + r43;
    r45 = r12 * r12;
    r46 = r4 * r33;
    r47 = r45 + r46;
    r48 = r42 + r47;
    r49 = r6 * r4;
    r49 = fmaf(r37, r49, r48 * r44);
    r49 = fmaf(r4, r23, r49);
    r23 = r6 * r4;
    r12 = r12 * r12;
    r12 = r12 * r4;
    r44 = r33 + r12;
    r42 = r42 + r44;
    r37 = r7 * r4;
    r48 = r10 * r11;
    r48 = r48 * r8;
    r36 = r48 + r36;
    r37 = fmaf(r36, r37, r42 * r23);
    r23 = r5 * r4;
    r10 = r9 * r10;
    r10 = r10 * r8;
    r19 = r19 + r10;
    r37 = fmaf(r19, r23, r37);
    r23 = fmaf(r31, r37, r3 * r49);
    r42 = r20 * r40;
    r50 = fmaf(r39, r37, r30 * r49);
    r51 = r20 * r29;
    r37 = fmaf(r32, r37, r18 * r49);
    r51 = fmaf(r37, r51, r50 * r42);
    r42 = r20 * r17;
    r51 = fmaf(r23, r42, r51);
    r49 = r17 * r51;
    r52 = -5.00000000000000000e-01;
    r14 = r52 * r14;
    r28 = rsqrtf(r28);
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r15 = rsqrtf(r15);
    r14 = r14 * r28;
    r14 = r14 * r27;
    r14 = r14 * r15;
    r49 = fmaf(r14, r49, r23 * r41);
    r23 = r17 * r49;
    r15 = r4 * r41;
    r27 = r29 * r14;
    r37 = fmaf(r51, r27, r37 * r41);
    r28 = r29 * r37;
    r28 = fmaf(r15, r28, r15 * r23);
    r23 = r40 * r15;
    r52 = r40 * r51;
    r52 = fmaf(r14, r52, r50 * r41);
    r28 = fmaf(r52, r23, r28);
    r50 = r7 * r4;
    r53 = r4 * r13;
    r54 = r16 + r53;
    r44 = r44 + r54;
    r55 = r6 * r4;
    r48 = r34 + r48;
    r55 = fmaf(r48, r55, r44 * r50);
    r50 = r5 * r4;
    r11 = r9 * r11;
    r11 = r11 * r8;
    r25 = r11 + r25;
    r55 = fmaf(r25, r50, r55);
    r50 = r5 * r4;
    r45 = r33 + r45;
    r45 = r45 + r43;
    r45 = r45 + r53;
    r22 = fmaf(r4, r22, r45 * r50);
    r22 = fmaf(r4, r26, r22);
    r26 = fmaf(r31, r22, r0 * r55);
    r50 = r20 * r29;
    r45 = fmaf(r32, r22, r1 * r55);
    r53 = r20 * r40;
    r22 = fmaf(r39, r22, r2 * r55);
    r53 = fmaf(r22, r53, r45 * r50);
    r53 = fmaf(r26, r42, r53);
    r50 = r17 * r53;
    r50 = fmaf(r14, r50, r26 * r41);
    r26 = r17 * r50;
    r55 = r40 * r53;
    r55 = fmaf(r14, r55, r22 * r41);
    r26 = fmaf(r55, r23, r15 * r26);
    r45 = fmaf(r45, r41, r53 * r27);
    r22 = r29 * r45;
    r26 = fmaf(r15, r22, r26);
    r22 = r5 * r4;
    r43 = r16 + r13;
    r43 = r43 + r46;
    r43 = r43 + r12;
    r12 = r6 * r4;
    r24 = r10 + r24;
    r12 = fmaf(r24, r12, r43 * r22);
    r22 = r7 * r4;
    r11 = r38 + r11;
    r12 = fmaf(r11, r22, r12);
    r22 = r6 * r4;
    r54 = r47 + r54;
    r47 = r7 * r4;
    r47 = fmaf(r35, r47, r54 * r22);
    r22 = r5 * r4;
    r47 = fmaf(r21, r22, r47);
    r22 = fmaf(r1, r47, r18 * r12);
    r21 = r20 * r40;
    r35 = fmaf(r2, r47, r30 * r12);
    r54 = r20 * r29;
    r54 = fmaf(r22, r54, r35 * r21);
    r47 = fmaf(r0, r47, r3 * r12);
    r54 = fmaf(r47, r42, r54);
    r22 = fmaf(r54, r27, r22 * r41);
    r12 = r29 * r22;
    r21 = r40 * r54;
    r35 = fmaf(r35, r41, r14 * r21);
    r12 = fmaf(r35, r23, r15 * r12);
    r21 = r17 * r54;
    r47 = fmaf(r47, r41, r14 * r21);
    r21 = r17 * r47;
    r12 = fmaf(r15, r21, r12);
    r21 = r20 * r40;
    r13 = r20 * r13;
    r16 = fmaf(r20, r16, r4);
    r38 = r13 + r16;
    r43 = fmaf(r39, r25, r2 * r38);
    r43 = fmaf(r30, r19, r43);
    r10 = r20 * r29;
    r46 = fmaf(r32, r25, r1 * r38);
    r46 = fmaf(r18, r19, r46);
    r10 = fmaf(r46, r10, r43 * r21);
    r25 = fmaf(r31, r25, r0 * r38);
    r25 = fmaf(r3, r19, r25);
    r10 = fmaf(r25, r42, r10);
    r19 = r17 * r10;
    r25 = fmaf(r25, r41, r14 * r19);
    r19 = r17 * r25;
    r38 = r40 * r10;
    r38 = fmaf(r14, r38, r43 * r41);
    r19 = fmaf(r38, r23, r15 * r19);
    r46 = fmaf(r10, r27, r46 * r41);
    r43 = r29 * r46;
    r19 = fmaf(r15, r43, r19);
    WriteSum4<float, float>((float*)inout_shared, r28, r26, r12, r19);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 + r13;
    r33 = r20 * r33;
    r13 = r13 + r33;
    r19 = fmaf(r2, r24, r30 * r13);
    r19 = fmaf(r39, r48, r19);
    r12 = r20 * r29;
    r26 = fmaf(r1, r24, r18 * r13);
    r26 = fmaf(r32, r48, r26);
    r28 = r20 * r40;
    r28 = fmaf(r19, r28, r26 * r12);
    r24 = fmaf(r0, r24, r3 * r13);
    r24 = fmaf(r31, r48, r24);
    r28 = fmaf(r24, r42, r28);
    r48 = r40 * r28;
    r48 = fmaf(r14, r48, r19 * r41);
    r26 = fmaf(r26, r41, r28 * r27);
    r19 = r29 * r26;
    r19 = fmaf(r15, r19, r48 * r23);
    r13 = r17 * r28;
    r24 = fmaf(r24, r41, r14 * r13);
    r13 = r17 * r24;
    r19 = fmaf(r15, r13, r19);
    r13 = r20 * r40;
    r16 = r33 + r16;
    r2 = fmaf(r2, r11, r39 * r16);
    r2 = fmaf(r30, r36, r2);
    r30 = r20 * r29;
    r1 = fmaf(r1, r11, r32 * r16);
    r1 = fmaf(r18, r36, r1);
    r30 = fmaf(r1, r30, r2 * r13);
    r11 = fmaf(r0, r11, r31 * r16);
    r11 = fmaf(r3, r36, r11);
    r30 = fmaf(r11, r42, r30);
    r42 = r40 * r30;
    r2 = fmaf(r2, r41, r14 * r42);
    r42 = r17 * r30;
    r42 = fmaf(r14, r42, r11 * r41);
    r11 = r17 * r42;
    r11 = fmaf(r15, r11, r2 * r23);
    r41 = fmaf(r1, r41, r30 * r27);
    r1 = r29 * r41;
    r11 = fmaf(r15, r1, r11);
    WriteSum2<float, float>((float*)inout_shared, r19, r11);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r52, r52, r49 * r49);
    r11 = fmaf(r37, r37, r11);
    r19 = fmaf(r50, r50, r45 * r45);
    r19 = fmaf(r55, r55, r19);
    r1 = fmaf(r22, r22, r35 * r35);
    r1 = fmaf(r47, r47, r1);
    r15 = fmaf(r38, r38, r46 * r46);
    r15 = fmaf(r25, r25, r15);
    WriteSum4<float, float>((float*)inout_shared, r11, r19, r1, r15);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r26, r26, r24 * r24);
    r15 = fmaf(r48, r48, r15);
    r1 = fmaf(r2, r2, r42 * r42);
    r1 = fmaf(r41, r41, r1);
    WriteSum2<float, float>((float*)inout_shared, r15, r1);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r37, r45, r52 * r55);
    r1 = fmaf(r49, r50, r1);
    r15 = fmaf(r49, r47, r52 * r35);
    r15 = fmaf(r37, r22, r15);
    r19 = fmaf(r52, r38, r37 * r46);
    r19 = fmaf(r49, r25, r19);
    r11 = fmaf(r52, r48, r37 * r26);
    r11 = fmaf(r49, r24, r11);
    WriteSum4<float, float>((float*)inout_shared, r1, r15, r19, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r49, r42, r52 * r2);
    r52 = fmaf(r37, r41, r52);
    r11 = fmaf(r45, r22, r50 * r47);
    r11 = fmaf(r55, r35, r11);
    r19 = fmaf(r50, r25, r55 * r38);
    r19 = fmaf(r45, r46, r19);
    r15 = fmaf(r50, r24, r45 * r26);
    r15 = fmaf(r55, r48, r15);
    WriteSum4<float, float>((float*)inout_shared, r52, r11, r19, r15);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r45, r41, r55 * r2);
    r55 = fmaf(r50, r42, r55);
    r15 = fmaf(r22, r46, r35 * r38);
    r15 = fmaf(r47, r25, r15);
    r19 = fmaf(r22, r26, r35 * r48);
    r19 = fmaf(r47, r24, r19);
    r11 = fmaf(r22, r41, r47 * r42);
    r11 = fmaf(r35, r2, r11);
    WriteSum4<float, float>((float*)inout_shared, r55, r15, r19, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r25, r24, r38 * r48);
    r11 = fmaf(r46, r26, r11);
    r38 = fmaf(r38, r2, r46 * r41);
    r38 = fmaf(r25, r42, r38);
    r2 = fmaf(r48, r2, r26 * r41);
    r2 = fmaf(r24, r42, r2);
    WriteSum3<float, float>((float*)inout_shared, r11, r38, r2);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
}

void FixedRigPositionPriorResJac(float* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 float* position,
                                 unsigned int position_num_alloc,
                                 float* sqrt_information,
                                 unsigned int sqrt_information_num_alloc,
                                 const float* const position_loss_scale,
                                 float* out_res,
                                 unsigned int out_res_num_alloc,
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
  FixedRigPositionPriorResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      position,
      position_num_alloc,
      sqrt_information,
      sqrt_information_num_alloc,
      position_loss_scale,
      out_res,
      out_res_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar