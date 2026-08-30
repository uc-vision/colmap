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
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* position,
    unsigned int position_num_alloc,
    double* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    const double* const position_loss_scale,
    double* out_res,
    unsigned int out_res_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;

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
    r18 = r11 * r12;
    r18 = r18 * r17;
    r19 = r16 + r18;
    r19 = r4 * r19;
    r15 = fma(r3, r15, r19);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = r11 * r6;
    r21 = r21 * r17;
    r22 = r7 * r5;
    r23 = r12 * r22;
    r24 = r21 + r23;
    r24 = r20 * r24;
    r15 = r15 + r24;
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r25, r26);
    r25 = fma(r25, r2, r2 * r15);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            6 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r27);
    r28 = r11 * r11;
    r29 = r5 * r28;
    r14 = r29 + r14;
    r30 = r12 * r6;
    r30 = r30 * r17;
    r31 = r11 * r22;
    r32 = r30 + r31;
    r14 = fma(r4, r32, r20 * r14);
    r33 = r12 * r7;
    r33 = r33 * r17;
    r21 = r21 + r33;
    r21 = r3 * r21;
    r14 = r14 + r21;
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r34);
    r34 = fma(r34, r2, r2 * r14);
    r14 = fma(r15, r34, r0 * r25);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            2 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r35,
                                            r36);
    r9 = r10 + r9;
    r9 = r9 + r29;
    r29 = r11 * r7;
    r29 = r29 * r17;
    r30 = r30 + r29;
    r9 = fma(r20, r30, r4 * r9);
    r22 = r6 * r22;
    r18 = r18 + r22;
    r9 = fma(r3, r18, r9);
    r26 = fma(r26, r2, r2 * r9);
    r14 = fma(r36, r26, r14);
  };
  LoadUnique<1, double, double>(position_loss_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = r9 * r9;
    r9 = 1.0 / r9;
    ReadIdx1<1024, double, double, double>(sqrt_information,
                                           8 * sqrt_information_num_alloc,
                                           global_thread_idx,
                                           r37);
    r38 = fma(r37, r34, r35 * r25);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            4 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r39,
                                            r40);
    r38 = fma(r40, r26, r38);
    r34 = fma(r27, r34, r1 * r25);
    r34 = fma(r39, r26, r34);
    r26 = fma(r34, r34, r38 * r38);
    r26 = fma(r14, r14, r26);
    r26 = fma(r26, r9, r10);
    r25 = sqrt(r26);
    r25 = r10 + r25;
    r10 = 1.0 / r25;
    r10 = r17 * r10;
    r41 = sqrt(r10);
    r42 = r14 * r41;
    r43 = r34 * r41;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r42, r43);
    r43 = r38 * r41;
    WriteIdx1<1024, double, double, double>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r43);
    r43 = r20 * r2;
    r42 = r2 * r13;
    r44 = r8 + r42;
    r45 = r7 * r7;
    r46 = r2 * r28;
    r47 = r45 + r46;
    r48 = r44 + r47;
    r49 = r4 * r2;
    r49 = fma(r32, r49, r48 * r43);
    r49 = fma(r2, r21, r49);
    r21 = r4 * r2;
    r7 = r7 * r7;
    r7 = r7 * r2;
    r43 = r28 + r7;
    r44 = r44 + r43;
    r32 = r20 * r2;
    r48 = r12 * r6;
    r48 = r48 * r5;
    r31 = r48 + r31;
    r32 = fma(r31, r32, r44 * r21);
    r21 = r3 * r2;
    r12 = r11 * r12;
    r12 = r12 * r5;
    r16 = r16 + r12;
    r32 = fma(r16, r21, r32);
    r21 = fma(r15, r32, r36 * r49);
    r44 = r17 * r38;
    r50 = fma(r37, r32, r40 * r49);
    r51 = r17 * r34;
    r32 = fma(r27, r32, r39 * r49);
    r51 = fma(r32, r51, r50 * r44);
    r44 = r17 * r14;
    r51 = fma(r21, r44, r51);
    r49 = r14 * r51;
    r52 = -5.00000000000000000e-01;
    r9 = r52 * r9;
    r26 = rsqrt(r26);
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r10 = rsqrt(r10);
    r9 = r9 * r26;
    r9 = r9 * r25;
    r9 = r9 * r10;
    r49 = fma(r9, r49, r21 * r41);
    r21 = r14 * r49;
    r10 = r2 * r41;
    r25 = r34 * r9;
    r32 = fma(r51, r25, r32 * r41);
    r26 = r34 * r32;
    r26 = fma(r10, r26, r10 * r21);
    r21 = r38 * r10;
    r52 = r38 * r51;
    r52 = fma(r9, r52, r50 * r41);
    r26 = fma(r52, r21, r26);
    r50 = r20 * r2;
    r53 = r2 * r8;
    r54 = r13 + r53;
    r43 = r43 + r54;
    r55 = r4 * r2;
    r48 = r29 + r48;
    r55 = fma(r48, r55, r43 * r50);
    r50 = r3 * r2;
    r6 = r11 * r6;
    r6 = r6 * r5;
    r23 = r6 + r23;
    r55 = fma(r23, r50, r55);
    r50 = r3 * r2;
    r45 = r28 + r45;
    r45 = r45 + r42;
    r45 = r45 + r53;
    r19 = fma(r2, r19, r45 * r50);
    r19 = fma(r2, r24, r19);
    r24 = fma(r15, r19, r0 * r55);
    r50 = r17 * r34;
    r45 = fma(r27, r19, r1 * r55);
    r53 = r17 * r38;
    r19 = fma(r37, r19, r35 * r55);
    r53 = fma(r19, r53, r45 * r50);
    r53 = fma(r24, r44, r53);
    r50 = r14 * r53;
    r50 = fma(r9, r50, r24 * r41);
    r24 = r14 * r50;
    r55 = r38 * r53;
    r55 = fma(r9, r55, r19 * r41);
    r24 = fma(r55, r21, r10 * r24);
    r45 = fma(r45, r41, r53 * r25);
    r19 = r34 * r45;
    r24 = fma(r10, r19, r24);
    WriteSum2<double, double>((double*)inout_shared, r26, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r3 * r2;
    r26 = r13 + r8;
    r26 = r26 + r46;
    r26 = r26 + r7;
    r7 = r4 * r2;
    r22 = r12 + r22;
    r7 = fma(r22, r7, r26 * r24);
    r24 = r20 * r2;
    r6 = r33 + r6;
    r7 = fma(r6, r24, r7);
    r24 = r4 * r2;
    r54 = r47 + r54;
    r47 = r20 * r2;
    r47 = fma(r30, r47, r54 * r24);
    r24 = r3 * r2;
    r47 = fma(r18, r24, r47);
    r24 = fma(r1, r47, r39 * r7);
    r18 = r17 * r38;
    r30 = fma(r35, r47, r40 * r7);
    r54 = r17 * r34;
    r54 = fma(r24, r54, r30 * r18);
    r47 = fma(r0, r47, r36 * r7);
    r54 = fma(r47, r44, r54);
    r24 = fma(r54, r25, r24 * r41);
    r7 = r34 * r24;
    r18 = r38 * r54;
    r30 = fma(r30, r41, r9 * r18);
    r7 = fma(r30, r21, r10 * r7);
    r18 = r14 * r54;
    r47 = fma(r47, r41, r9 * r18);
    r18 = r14 * r47;
    r7 = fma(r10, r18, r7);
    r18 = r17 * r38;
    r8 = r17 * r8;
    r13 = fma(r17, r13, r2);
    r33 = r8 + r13;
    r26 = fma(r37, r23, r35 * r33);
    r26 = fma(r40, r16, r26);
    r12 = r17 * r34;
    r46 = fma(r27, r23, r1 * r33);
    r46 = fma(r39, r16, r46);
    r12 = fma(r46, r12, r26 * r18);
    r23 = fma(r15, r23, r0 * r33);
    r23 = fma(r36, r16, r23);
    r12 = fma(r23, r44, r12);
    r16 = r14 * r12;
    r23 = fma(r23, r41, r9 * r16);
    r16 = r14 * r23;
    r33 = r38 * r12;
    r33 = fma(r9, r33, r26 * r41);
    r16 = fma(r33, r21, r10 * r16);
    r46 = fma(r12, r25, r46 * r41);
    r26 = r34 * r46;
    r16 = fma(r10, r26, r16);
    WriteSum2<double, double>((double*)inout_shared, r7, r16);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r2 + r8;
    r28 = r17 * r28;
    r8 = r8 + r28;
    r16 = fma(r35, r22, r40 * r8);
    r16 = fma(r37, r48, r16);
    r7 = r17 * r34;
    r26 = fma(r1, r22, r39 * r8);
    r26 = fma(r27, r48, r26);
    r18 = r17 * r38;
    r18 = fma(r16, r18, r26 * r7);
    r22 = fma(r0, r22, r36 * r8);
    r22 = fma(r15, r48, r22);
    r18 = fma(r22, r44, r18);
    r48 = r38 * r18;
    r48 = fma(r9, r48, r16 * r41);
    r26 = fma(r26, r41, r18 * r25);
    r16 = r34 * r26;
    r16 = fma(r10, r16, r48 * r21);
    r8 = r14 * r18;
    r22 = fma(r22, r41, r9 * r8);
    r8 = r14 * r22;
    r16 = fma(r10, r8, r16);
    r8 = r17 * r38;
    r13 = r28 + r13;
    r35 = fma(r35, r6, r37 * r13);
    r35 = fma(r40, r31, r35);
    r40 = r17 * r34;
    r1 = fma(r1, r6, r27 * r13);
    r1 = fma(r39, r31, r1);
    r40 = fma(r1, r40, r35 * r8);
    r6 = fma(r0, r6, r15 * r13);
    r6 = fma(r36, r31, r6);
    r40 = fma(r6, r44, r40);
    r44 = r38 * r40;
    r35 = fma(r35, r41, r9 * r44);
    r44 = r14 * r40;
    r44 = fma(r9, r44, r6 * r41);
    r6 = r14 * r44;
    r6 = fma(r10, r6, r35 * r21);
    r41 = fma(r1, r41, r40 * r25);
    r1 = r34 * r41;
    r6 = fma(r10, r1, r6);
    WriteSum2<double, double>((double*)inout_shared, r16, r6);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r52, r52, r49 * r49);
    r6 = fma(r32, r32, r6);
    r16 = fma(r50, r50, r45 * r45);
    r16 = fma(r55, r55, r16);
    WriteSum2<double, double>((double*)inout_shared, r6, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r24, r24, r30 * r30);
    r16 = fma(r47, r47, r16);
    r6 = fma(r33, r33, r46 * r46);
    r6 = fma(r23, r23, r6);
    WriteSum2<double, double>((double*)inout_shared, r16, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r26, r26, r22 * r22);
    r6 = fma(r48, r48, r6);
    r16 = fma(r35, r35, r44 * r44);
    r16 = fma(r41, r41, r16);
    WriteSum2<double, double>((double*)inout_shared, r6, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r32, r45, r52 * r55);
    r16 = fma(r49, r50, r16);
    r6 = fma(r49, r47, r52 * r30);
    r6 = fma(r32, r24, r6);
    WriteSum2<double, double>((double*)inout_shared, r16, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r52, r33, r32 * r46);
    r6 = fma(r49, r23, r6);
    r16 = fma(r52, r48, r32 * r26);
    r16 = fma(r49, r22, r16);
    WriteSum2<double, double>((double*)inout_shared, r6, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r49, r44, r52 * r35);
    r52 = fma(r32, r41, r52);
    r16 = fma(r45, r24, r50 * r47);
    r16 = fma(r55, r30, r16);
    WriteSum2<double, double>((double*)inout_shared, r52, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r50, r23, r55 * r33);
    r16 = fma(r45, r46, r16);
    r52 = fma(r50, r22, r45 * r26);
    r52 = fma(r55, r48, r52);
    WriteSum2<double, double>((double*)inout_shared, r16, r52);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r45, r41, r55 * r35);
    r55 = fma(r50, r44, r55);
    r52 = fma(r24, r46, r30 * r33);
    r52 = fma(r47, r23, r52);
    WriteSum2<double, double>((double*)inout_shared, r55, r52);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r24, r26, r30 * r48);
    r52 = fma(r47, r22, r52);
    r55 = fma(r24, r41, r47 * r44);
    r55 = fma(r30, r35, r55);
    WriteSum2<double, double>((double*)inout_shared, r52, r55);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r23, r22, r33 * r48);
    r55 = fma(r46, r26, r55);
    r33 = fma(r33, r35, r46 * r41);
    r33 = fma(r23, r44, r33);
    WriteSum2<double, double>((double*)inout_shared, r55, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r48, r35, r26 * r41);
    r35 = fma(r22, r44, r35);
    WriteSum1<double, double>((double*)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
}

void FixedRigPositionPriorResJac(double* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 double* position,
                                 unsigned int position_num_alloc,
                                 double* sqrt_information,
                                 unsigned int sqrt_information_num_alloc,
                                 const double* const position_loss_scale,
                                 double* out_res,
                                 unsigned int out_res_num_alloc,
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