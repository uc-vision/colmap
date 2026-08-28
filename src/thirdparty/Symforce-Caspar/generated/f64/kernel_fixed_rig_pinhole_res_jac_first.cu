#include "kernel_fixed_rig_pinhole_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedRigPinholeResJacFirstKernel(
    double* pose,
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
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    double* const out_sensor_from_rig_log_scale_precond_diag,
    double* const out_sensor_from_rig_log_scale_precond_tril,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
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

  __shared__ double out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ double out_sensor_from_rig_log_scale_precond_diag_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib, 2 * calib_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = 1.00000000000000008e-15;
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r5, r6);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10);
    r11 = r7 * r10;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    r16 = fma(r13, r14, r11);
    r17 = r12 * r15;
    r16 = fma(r8, r9, r16);
    r16 = fma(r4, r17, r16);
    r18 = 2.00000000000000000e+00;
    r19 = r16 * r18;
    r20 = fma(r8, r14, r12 * r10);
    r21 = r13 * r9;
    r20 = fma(r4, r21, r20);
    r20 = fma(r7, r15, r20);
    r19 = r19 * r20;
    r21 = r7 * r14;
    r21 = fma(r4, r21, r13 * r10);
    r21 = fma(r8, r15, r21);
    r21 = fma(r12, r9, r21);
    r22 = -2.00000000000000000e+00;
    r23 = fma(r13, r15, r12 * r14);
    r23 = fma(r7, r9, r23);
    r23 = fma(r4, r23, r8 * r10);
    r24 = r22 * r23;
    r25 = fma(r21, r24, r19);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r26, r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r14 * r9;
    r28 = r28 * r18;
    r29 = r15 * r10;
    r29 = fma(r22, r29, r28);
    r30 = fma(r26, r29, r5 * r25);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r14 * r14;
    r32 = r22 * r32;
    r33 = 1.00000000000000000e+00;
    r34 = r15 * r15;
    r34 = fma(r22, r34, r33);
    r35 = r32 + r34;
    r36 = r15 * r9;
    r36 = r36 * r18;
    r37 = r14 * r10;
    r37 = fma(r18, r37, r36);
    r38 = r18 * r20;
    r39 = r18 * r21;
    r40 = r16 * r39;
    r38 = fma(r23, r38, r40);
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r41);
  };
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = 2.71828182845904523536;
    r42 = pow(r43, r42);
    r41 = r41 * r42;
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r43);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = r22 * r21;
    r44 = r44 * r21;
    r45 = r33 + r44;
    r46 = r22 * r20;
    r46 = r46 * r20;
    r45 = r45 + r46;
    r30 = fma(r31, r35, r30);
    r30 = fma(r27, r37, r30);
    r30 = fma(r6, r38, r30);
    r30 = r30 + r41;
    r30 = fma(r43, r45, r30);
    r47 = copysign(1.0, r30);
    r47 = fma(r0, r47, r30);
    r0 = 1.0 / r47;
    ReadIdx2<1024, double, double, double2>(
        calib, 0 * calib_num_alloc, global_thread_idx, r30, r48);
    r49 = r22 * r16;
    r49 = r49 * r16;
    r50 = r33 + r49;
    r50 = r50 + r44;
    r44 = r20 * r39;
    r51 = fma(r16, r24, r44);
    r52 = fma(r6, r51, r5 * r50);
    r19 = fma(r23, r39, r19);
    r53 = r15 * r10;
    r53 = fma(r18, r53, r28);
    r28 = r9 * r10;
    r54 = r14 * r15;
    r54 = r54 * r18;
    r28 = fma(r22, r28, r54);
    r55 = r9 * r9;
    r55 = r22 * r55;
    r34 = r55 + r34;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r56,
                                            r57);
    r52 = fma(r43, r19, r52);
    r52 = fma(r31, r53, r52);
    r52 = fma(r27, r28, r52);
    r52 = fma(r26, r34, r52);
    r52 = fma(r56, r42, r52);
    r52 = r30 * r52;
    r2 = fma(r0, r52, r2);
    r3 = fma(r3, r4, r1);
    r1 = r16 * r18;
    r1 = fma(r23, r1, r44);
    r44 = r9 * r10;
    r44 = fma(r18, r44, r54);
    r26 = fma(r26, r44, r5 * r1);
    r55 = r33 + r55;
    r55 = r55 + r32;
    r32 = r14 * r10;
    r32 = fma(r22, r32, r36);
    r40 = fma(r20, r24, r40);
    r49 = r33 + r49;
    r49 = r49 + r46;
    r26 = fma(r27, r55, r26);
    r26 = fma(r31, r32, r26);
    r26 = fma(r43, r40, r26);
    r26 = fma(r57, r42, r26);
    r26 = fma(r6, r49, r26);
    r26 = r48 * r26;
    r3 = fma(r0, r26, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r31 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r31);
  if (global_thread_idx < problem_size) {
    r31 = r18 * r23;
    r27 = r13 * r14;
    r46 = 5.00000000000000000e-01;
    r27 = fma(r46, r11, r46 * r27);
    r33 = -5.00000000000000000e-01;
    r36 = r8 * r46;
    r27 = fma(r33, r17, r27);
    r27 = fma(r9, r36, r27);
    r54 = r12 * r10;
    r58 = r8 * r14;
    r58 = fma(r33, r58, r33 * r54);
    r54 = r7 * r15;
    r58 = fma(r33, r54, r58);
    r59 = r13 * r9;
    r58 = fma(r46, r59, r58);
    r31 = fma(r58, r39, r27 * r31);
    r59 = r18 * r20;
    r54 = r7 * r14;
    r60 = r8 * r15;
    r60 = fma(r33, r60, r46 * r54);
    r54 = r12 * r9;
    r60 = fma(r33, r54, r60);
    r61 = r13 * r33;
    r60 = fma(r10, r61, r60);
    r54 = r16 * r18;
    r62 = r12 * r14;
    r63 = r7 * r9;
    r63 = fma(r33, r63, r33 * r62);
    r63 = fma(r10, r36, r63);
    r63 = fma(r15, r61, r63);
    r54 = r54 * r63;
    r59 = fma(r60, r59, r54);
    r31 = r31 + r59;
    r62 = r18 * r20;
    r62 = r62 * r27;
    r64 = r22 * r16;
    r64 = fma(r58, r64, r62);
    r65 = r63 * r39;
    r64 = r64 + r65;
    r64 = fma(r60, r24, r64);
    r64 = fma(r6, r64, r43 * r31);
    r31 = r21 * r27;
    r66 = -4.00000000000000000e+00;
    r31 = r31 * r66;
    r67 = r16 * r60;
    r68 = r66 * r67;
    r69 = r31 + r68;
    r64 = fma(r5, r69, r64);
    r69 = r30 * r64;
    r70 = r16 * r18;
    r71 = r60 * r39;
    r70 = fma(r27, r70, r71);
    r72 = r18 * r20;
    r72 = r72 * r58;
    r73 = r18 * r23;
    r73 = r73 * r63;
    r74 = r72 + r73;
    r75 = r70 + r74;
    r76 = r22 * r21;
    r27 = fma(r27, r24, r58 * r76);
    r27 = r27 + r59;
    r27 = fma(r5, r27, r6 * r75);
    r75 = r20 * r66;
    r76 = r63 * r75;
    r31 = r31 + r76;
    r27 = fma(r43, r31, r27);
    r47 = r47 * r47;
    r47 = 1.0 / r47;
    r47 = r4 * r47;
    r52 = r47 * r52;
    r69 = fma(r27, r52, r0 * r69);
    r31 = r22 * r20;
    r77 = r63 * r24;
    r31 = fma(r58, r31, r77);
    r31 = r31 + r70;
    r76 = r68 + r76;
    r76 = fma(r6, r76, r43 * r31);
    r31 = r18 * r23;
    r31 = fma(r60, r31, r62);
    r62 = r16 * r18;
    r62 = fma(r58, r62, r65);
    r31 = r31 + r62;
    r76 = fma(r5, r31, r76);
    r31 = r48 * r76;
    r65 = r27 * r47;
    r65 = fma(r26, r65, r0 * r31);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r69, r65);
    r31 = r22 * r21;
    r31 = fma(r60, r31, r77);
    r68 = r16 * r18;
    r70 = r8 * r9;
    r11 = fma(r33, r11, r33 * r70);
    r11 = fma(r14, r61, r11);
    r11 = fma(r46, r17, r11);
    r68 = r68 * r11;
    r17 = r18 * r20;
    r70 = r12 * r10;
    r78 = r7 * r15;
    r78 = fma(r46, r78, r46 * r70);
    r78 = fma(r14, r36, r78);
    r78 = fma(r9, r61, r78);
    r17 = fma(r78, r17, r68);
    r31 = r31 + r17;
    r61 = r18 * r23;
    r70 = r78 * r39;
    r61 = fma(r11, r61, r70);
    r61 = r61 + r59;
    r61 = fma(r6, r61, r5 * r31);
    r31 = r21 * r63;
    r31 = r31 * r66;
    r59 = r11 * r75;
    r79 = r31 + r59;
    r61 = fma(r43, r79, r61);
    r71 = r73 + r71;
    r71 = r71 + r17;
    r17 = r16 * r66;
    r17 = r17 * r78;
    r31 = r31 + r17;
    r31 = fma(r5, r31, r43 * r71);
    r71 = fma(r78, r24, r22 * r67);
    r73 = r18 * r20;
    r73 = r73 * r63;
    r79 = fma(r11, r39, r73);
    r71 = r71 + r79;
    r31 = fma(r6, r71, r31);
    r71 = r30 * r31;
    r71 = fma(r0, r71, r61 * r52);
    r80 = r22 * r20;
    r80 = fma(r60, r80, r54);
    r80 = r80 + r70;
    r80 = fma(r11, r24, r80);
    r70 = r18 * r23;
    r67 = fma(r18, r67, r78 * r70);
    r67 = r67 + r79;
    r67 = fma(r5, r67, r43 * r80);
    r59 = r17 + r59;
    r67 = fma(r6, r59, r67);
    r59 = r48 * r67;
    r17 = r61 * r47;
    r17 = fma(r26, r17, r0 * r59);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r71, r17);
    r59 = r22 * r16;
    r59 = fma(r11, r59, r72);
    r72 = r13 * r10;
    r80 = r7 * r14;
    r80 = fma(r33, r80, r46 * r72);
    r72 = r12 * r9;
    r80 = fma(r46, r72, r80);
    r80 = fma(r15, r36, r80);
    r39 = r80 * r39;
    r59 = r59 + r77;
    r59 = r59 + r39;
    r63 = r16 * r63;
    r63 = r63 * r66;
    r77 = r21 * r58;
    r77 = r77 * r66;
    r66 = r63 + r77;
    r66 = fma(r5, r66, r6 * r59);
    r59 = r16 * r18;
    r59 = r59 * r80;
    r36 = r18 * r23;
    r36 = fma(r58, r36, r59);
    r36 = r36 + r79;
    r66 = fma(r43, r36, r66);
    r36 = r30 * r66;
    r75 = r80 * r75;
    r77 = r77 + r75;
    r59 = r73 + r59;
    r73 = r22 * r21;
    r59 = fma(r11, r73, r59);
    r59 = fma(r58, r24, r59);
    r59 = fma(r5, r59, r43 * r77);
    r77 = r18 * r20;
    r58 = r18 * r23;
    r58 = fma(r80, r58, r11 * r77);
    r58 = r58 + r62;
    r59 = fma(r6, r58, r59);
    r36 = fma(r59, r52, r0 * r36);
    r58 = r59 * r47;
    r39 = r68 + r39;
    r39 = r39 + r74;
    r74 = r22 * r20;
    r24 = fma(r80, r24, r11 * r74);
    r24 = r24 + r62;
    r24 = fma(r43, r24, r5 * r39);
    r75 = r63 + r75;
    r24 = fma(r6, r75, r24);
    r75 = r48 * r24;
    r75 = fma(r0, r75, r26 * r58);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r36, r75);
    r58 = r30 * r34;
    r58 = fma(r0, r58, r29 * r52);
    r6 = r29 * r47;
    r63 = r48 * r44;
    r63 = fma(r0, r63, r26 * r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r58, r63);
    r6 = r30 * r28;
    r6 = fma(r0, r6, r37 * r52);
    r43 = r48 * r55;
    r39 = r37 * r47;
    r39 = fma(r26, r39, r0 * r43);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r6, r39);
    r43 = r30 * r53;
    r43 = fma(r0, r43, r35 * r52);
    r5 = r35 * r47;
    r62 = r48 * r32;
    r62 = fma(r0, r62, r26 * r5);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r43, r62);
    r5 = r4 * r3;
    r80 = r4 * r2;
    r80 = fma(r69, r80, r65 * r5);
    r5 = r4 * r2;
    r74 = r4 * r3;
    r74 = fma(r17, r74, r71 * r5);
    WriteSum2<double, double>((double*)inout_shared, r80, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r4 * r3;
    r80 = r4 * r2;
    r80 = fma(r36, r80, r75 * r74);
    r74 = r4 * r3;
    r5 = r4 * r2;
    r5 = fma(r58, r5, r63 * r74);
    WriteSum2<double, double>((double*)inout_shared, r80, r5);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r4 * r3;
    r80 = r4 * r2;
    r80 = fma(r6, r80, r39 * r5);
    r5 = r4 * r2;
    r74 = r4 * r3;
    r74 = fma(r62, r74, r43 * r5);
    WriteSum2<double, double>((double*)inout_shared, r80, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r69, r69, r65 * r65);
    r80 = fma(r71, r71, r17 * r17);
    WriteSum2<double, double>((double*)inout_shared, r74, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r36, r36, r75 * r75);
    r74 = fma(r58, r58, r63 * r63);
    WriteSum2<double, double>((double*)inout_shared, r80, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r39, r39, r6 * r6);
    r80 = fma(r62, r62, r43 * r43);
    WriteSum2<double, double>((double*)inout_shared, r74, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r65, r17, r69 * r71);
    r74 = fma(r69, r36, r65 * r75);
    WriteSum2<double, double>((double*)inout_shared, r80, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r69, r58, r65 * r63);
    r80 = fma(r69, r6, r65 * r39);
    WriteSum2<double, double>((double*)inout_shared, r74, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r65, r62, r69 * r43);
    r69 = fma(r71, r36, r17 * r75);
    WriteSum2<double, double>((double*)inout_shared, r65, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r71, r58, r17 * r63);
    r65 = fma(r71, r6, r17 * r39);
    WriteSum2<double, double>((double*)inout_shared, r69, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r17, r62, r71 * r43);
    r71 = fma(r36, r58, r75 * r63);
    WriteSum2<double, double>((double*)inout_shared, r17, r71);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fma(r36, r6, r75 * r39);
    r36 = fma(r36, r43, r75 * r62);
    WriteSum2<double, double>((double*)inout_shared, r71, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r63, r39, r58 * r6);
    r63 = fma(r63, r62, r58 * r43);
    WriteSum2<double, double>((double*)inout_shared, r36, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r6, r43, r39 * r62);
    WriteSum1<double, double>((double*)inout_shared, r43);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r56 * r30;
    r43 = r43 * r42;
    r43 = fma(r0, r43, r52 * r41);
    r6 = r47 * r26;
    r62 = r57 * r48;
    r62 = r62 * r42;
    r62 = fma(r0, r62, r41 * r6);
    WriteIdx2<1024, double, double, double2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r43,
        r62);
    r6 = r4 * r3;
    r41 = r4 * r2;
    r41 = fma(r43, r41, r62 * r6);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r41);
  if (global_thread_idx < problem_size) {
    r43 = fma(r43, r43, r62 * r62);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r43);
  if (global_thread_idx < problem_size) {
    r43 = r30 * r50;
    r43 = fma(r0, r43, r25 * r52);
    r62 = r25 * r47;
    r41 = r48 * r1;
    r41 = fma(r0, r41, r26 * r62);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r43,
                                             r41);
    r62 = r30 * r51;
    r62 = fma(r38, r52, r0 * r62);
    r6 = r38 * r47;
    r42 = r48 * r49;
    r42 = fma(r0, r42, r26 * r6);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r62,
                                             r42);
    r6 = r30 * r19;
    r6 = fma(r0, r6, r45 * r52);
    r52 = r48 * r40;
    r39 = r45 * r47;
    r39 = fma(r26, r39, r0 * r52);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r6, r39);
    r52 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r41, r0, r43 * r52);
    r52 = r4 * r3;
    r63 = r4 * r2;
    r63 = fma(r62, r63, r42 * r52);
    WriteSum2<double, double>((double*)inout_shared, r0, r63);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r39, r0, r6 * r63);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r43, r43, r41 * r41);
    r63 = fma(r42, r42, r62 * r62);
    WriteSum2<double, double>((double*)inout_shared, r0, r63);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r39, r39, r6 * r6);
    WriteSum1<double, double>((double*)inout_shared, r63);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r43, r62, r41 * r42);
    r43 = fma(r43, r6, r41 * r39);
    WriteSum2<double, double>((double*)inout_shared, r63, r43);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r62, r6, r42 * r39);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_njtr_local,
                        out_sensor_from_rig_log_scale_njtr,
                        1);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                        out_sensor_from_rig_log_scale_precond_diag,
                        1);
}

void FixedRigPinholeResJacFirst(
    double* pose,
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
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    double* const out_sensor_from_rig_log_scale_precond_diag,
    double* const out_sensor_from_rig_log_scale_precond_tril,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigPinholeResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
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
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_sensor_from_rig_log_scale_jac,
      out_sensor_from_rig_log_scale_jac_num_alloc,
      out_sensor_from_rig_log_scale_njtr,
      out_sensor_from_rig_log_scale_precond_diag,
      out_sensor_from_rig_log_scale_precond_tril,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar