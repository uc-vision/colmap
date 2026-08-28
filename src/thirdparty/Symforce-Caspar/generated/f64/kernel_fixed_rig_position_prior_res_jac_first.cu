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
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* position,
        unsigned int position_num_alloc,
        double* sqrt_information,
        unsigned int sqrt_information_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            0 * sqrt_information_num_alloc,
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
    r5 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = r6 * r6;
    r9 = r5 * r8;
    r10 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r11, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = r12 * r12;
    r14 = fma(r5, r13, r10);
    r15 = r9 + r14;
    r16 = r6 * r7;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r11 * r17;
    r19 = r12 * r18;
    r20 = r16 + r19;
    r20 = r4 * r20;
    r15 = fma(r3, r15, r20);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r21);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r22 = r6 * r18;
    r23 = r7 * r5;
    r24 = r12 * r23;
    r25 = r22 + r24;
    r15 = fma(r21, r25, r15);
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r26, r27);
    r26 = fma(r26, r2, r2 * r15);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            6 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r28);
    r29 = r11 * r11;
    r29 = r29 * r5;
    r14 = r29 + r14;
    r30 = r12 * r6;
    r30 = r30 * r17;
    r31 = r11 * r23;
    r32 = r30 + r31;
    r14 = fma(r4, r32, r21 * r14);
    r33 = r12 * r7;
    r33 = r33 * r17;
    r22 = r33 + r22;
    r22 = r3 * r22;
    r14 = r14 + r22;
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r34);
    r34 = fma(r34, r2, r2 * r14);
    r14 = fma(r15, r34, r0 * r26);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            2 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r35,
                                            r36);
    r9 = r10 + r9;
    r9 = r9 + r29;
    r29 = r7 * r18;
    r30 = r30 + r29;
    r9 = fma(r21, r30, r4 * r9);
    r23 = r6 * r23;
    r19 = r19 + r23;
    r9 = fma(r3, r19, r9);
    r27 = fma(r27, r2, r2 * r9);
    r14 = fma(r36, r27, r14);
    r9 = fma(r28, r34, r1 * r26);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            4 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r37);
    r9 = fma(r10, r27, r9);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r14, r9);
    ReadIdx1<1024, double, double, double>(sqrt_information,
                                           8 * sqrt_information_num_alloc,
                                           global_thread_idx,
                                           r38);
    r34 = fma(r38, r34, r35 * r26);
    r34 = fma(r37, r27, r34);
    WriteIdx1<1024, double, double, double>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r34);
    r27 = fma(r9, r9, r34 * r34);
    r27 = fma(r14, r14, r27);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r27);
  if (global_thread_idx < problem_size) {
    r27 = r2 * r14;
    r26 = r4 * r2;
    r21 = r21 * r2;
    r39 = r2 * r13;
    r40 = r8 + r39;
    r41 = r7 * r7;
    r42 = r11 * r11;
    r42 = r42 * r2;
    r43 = r41 + r42;
    r44 = r40 + r43;
    r44 = fma(r44, r21, r32 * r26);
    r44 = fma(r2, r22, r44);
    r22 = r4 * r2;
    r26 = r11 * r11;
    r7 = r7 * r7;
    r7 = r7 * r2;
    r32 = r26 + r7;
    r40 = r40 + r32;
    r45 = r3 * r2;
    r46 = r11 * r12;
    r46 = r46 * r5;
    r16 = r16 + r46;
    r45 = fma(r16, r45, r40 * r22);
    r12 = r12 * r6;
    r12 = r12 * r5;
    r31 = r12 + r31;
    r45 = fma(r31, r21, r45);
    r22 = fma(r15, r45, r36 * r44);
    r40 = r2 * r34;
    r47 = fma(r38, r45, r37 * r44);
    r40 = fma(r47, r40, r22 * r27);
    r27 = r2 * r9;
    r45 = fma(r28, r45, r10 * r44);
    r40 = fma(r45, r27, r40);
    r27 = r2 * r9;
    r44 = r4 * r2;
    r29 = r12 + r29;
    r12 = r3 * r2;
    r6 = r11 * r6;
    r6 = r6 * r5;
    r24 = r6 + r24;
    r12 = fma(r24, r12, r29 * r44);
    r44 = r2 * r8;
    r5 = r13 + r44;
    r32 = r32 + r5;
    r12 = fma(r32, r21, r12);
    r32 = r3 * r2;
    r41 = r26 + r41;
    r41 = r41 + r39;
    r41 = r41 + r44;
    r20 = fma(r2, r20, r41 * r32);
    r20 = fma(r25, r21, r20);
    r25 = fma(r28, r20, r1 * r12);
    r32 = r2 * r34;
    r41 = fma(r38, r20, r35 * r12);
    r32 = fma(r41, r32, r25 * r27);
    r27 = r2 * r14;
    r20 = fma(r15, r20, r0 * r12);
    r32 = fma(r20, r27, r32);
    WriteSum2<double, double>((double*)inout_shared, r40, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r2 * r34;
    r40 = r3 * r2;
    r27 = r13 + r8;
    r27 = r27 + r42;
    r27 = r27 + r7;
    r7 = r4 * r2;
    r23 = r46 + r23;
    r7 = fma(r23, r7, r27 * r40);
    r6 = r33 + r6;
    r7 = fma(r6, r21, r7);
    r33 = r4 * r2;
    r5 = r43 + r5;
    r43 = r3 * r2;
    r43 = fma(r19, r43, r5 * r33);
    r43 = fma(r30, r21, r43);
    r21 = fma(r35, r43, r37 * r7);
    r30 = r2 * r14;
    r33 = fma(r0, r43, r36 * r7);
    r30 = fma(r33, r30, r21 * r32);
    r32 = r2 * r9;
    r43 = fma(r1, r43, r10 * r7);
    r30 = fma(r43, r32, r30);
    r32 = r2 * r14;
    r8 = r17 * r8;
    r13 = fma(r17, r13, r2);
    r17 = r8 + r13;
    r7 = fma(r15, r24, r0 * r17);
    r7 = fma(r36, r16, r7);
    r19 = r2 * r34;
    r5 = fma(r38, r24, r35 * r17);
    r5 = fma(r37, r16, r5);
    r19 = fma(r5, r19, r7 * r32);
    r32 = r2 * r9;
    r24 = fma(r28, r24, r1 * r17);
    r24 = fma(r10, r16, r24);
    r19 = fma(r24, r32, r19);
    WriteSum2<double, double>((double*)inout_shared, r30, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r2 * r9;
    r8 = r2 + r8;
    r18 = r11 * r18;
    r8 = r8 + r18;
    r11 = fma(r1, r23, r10 * r8);
    r11 = fma(r28, r29, r11);
    r30 = r2 * r14;
    r32 = fma(r0, r23, r36 * r8);
    r32 = fma(r15, r29, r32);
    r30 = fma(r32, r30, r11 * r19);
    r19 = r2 * r34;
    r23 = fma(r35, r23, r37 * r8);
    r23 = fma(r38, r29, r23);
    r30 = fma(r23, r19, r30);
    r19 = r2 * r34;
    r13 = r18 + r13;
    r35 = fma(r35, r6, r38 * r13);
    r35 = fma(r37, r31, r35);
    r37 = r2 * r9;
    r1 = fma(r1, r6, r28 * r13);
    r1 = fma(r10, r31, r1);
    r37 = fma(r1, r37, r35 * r19);
    r19 = r2 * r14;
    r6 = fma(r0, r6, r15 * r13);
    r6 = fma(r36, r31, r6);
    r37 = fma(r6, r19, r37);
    WriteSum2<double, double>((double*)inout_shared, r30, r37);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r45, r45, r47 * r47);
    r37 = fma(r22, r22, r37);
    r30 = fma(r20, r20, r41 * r41);
    r30 = fma(r25, r25, r30);
    WriteSum2<double, double>((double*)inout_shared, r37, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r21, r21, r33 * r33);
    r30 = fma(r43, r43, r30);
    r37 = fma(r5, r5, r24 * r24);
    r37 = fma(r7, r7, r37);
    WriteSum2<double, double>((double*)inout_shared, r30, r37);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r23, r23, r11 * r11);
    r37 = fma(r32, r32, r37);
    r30 = fma(r35, r35, r6 * r6);
    r30 = fma(r1, r1, r30);
    WriteSum2<double, double>((double*)inout_shared, r37, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r47, r41, r22 * r20);
    r30 = fma(r45, r25, r30);
    r37 = fma(r45, r43, r47 * r21);
    r37 = fma(r22, r33, r37);
    WriteSum2<double, double>((double*)inout_shared, r30, r37);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r47, r5, r45 * r24);
    r37 = fma(r22, r7, r37);
    r30 = fma(r47, r23, r45 * r11);
    r30 = fma(r22, r32, r30);
    WriteSum2<double, double>((double*)inout_shared, r37, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r47, r35, r22 * r6);
    r47 = fma(r45, r1, r47);
    r45 = fma(r41, r21, r20 * r33);
    r45 = fma(r25, r43, r45);
    WriteSum2<double, double>((double*)inout_shared, r47, r45);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r41, r5, r25 * r24);
    r45 = fma(r20, r7, r45);
    r47 = fma(r20, r32, r25 * r11);
    r47 = fma(r41, r23, r47);
    WriteSum2<double, double>((double*)inout_shared, r45, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r35, r20 * r6);
    r41 = fma(r25, r1, r41);
    r25 = fma(r33, r7, r21 * r5);
    r25 = fma(r43, r24, r25);
    WriteSum2<double, double>((double*)inout_shared, r41, r25);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r21, r23, r43 * r11);
    r25 = fma(r33, r32, r25);
    r21 = fma(r21, r35, r43 * r1);
    r21 = fma(r33, r6, r21);
    WriteSum2<double, double>((double*)inout_shared, r25, r21);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r5, r23, r7 * r32);
    r21 = fma(r24, r11, r21);
    r5 = fma(r5, r35, r24 * r1);
    r5 = fma(r7, r6, r5);
    WriteSum2<double, double>((double*)inout_shared, r21, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r23, r35, r32 * r6);
    r35 = fma(r11, r1, r35);
    WriteSum1<double, double>((double*)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedRigPositionPriorResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* position,
    unsigned int position_num_alloc,
    double* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
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