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
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* calib,
    unsigned int calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    const float* const reprojection_loss_scale,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    float* const out_sensor_from_rig_log_scale_njtr,
    float* const out_sensor_from_rig_log_scale_precond_diag,
    float* const out_sensor_from_rig_log_scale_precond_tril,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
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

  __shared__ float out_rTr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_precond_diag_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = fmaf(r14, r15, r11 * r18);
    r20 = r12 * r17;
    r19 = fmaf(r6, r20, r19);
    r19 = fmaf(r13, r16, r19);
    r20 = r10 * r19;
    r21 = r13 * r18;
    r22 = r12 * r15;
    r23 = r21 + r22;
    r24 = r11 * r16;
    r23 = fmaf(r14, r17, r23);
    r23 = fmaf(r6, r24, r23);
    r20 = r20 * r23;
    r25 = r13 * r15;
    r25 = fmaf(r6, r25, r12 * r18);
    r25 = fmaf(r14, r16, r25);
    r25 = fmaf(r11, r17, r25);
    r26 = fmaf(r12, r16, r11 * r15);
    r26 = fmaf(r13, r17, r26);
    r26 = fmaf(r6, r26, r14 * r18);
    r27 = -2.00000000000000000e+00;
    r28 = r26 * r27;
    r29 = fmaf(r25, r28, r20);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r30,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r15 * r17;
    r33 = r33 * r10;
    r34 = r16 * r18;
    r34 = fmaf(r27, r34, r33);
    r35 = fmaf(r30, r34, r7 * r29);
    r36 = r15 * r15;
    r36 = r36 * r27;
    r37 = 1.00000000000000000e+00;
    r38 = r16 * r16;
    r38 = fmaf(r27, r38, r37);
    r39 = r36 + r38;
    r40 = r16 * r17;
    r40 = r40 * r10;
    r41 = r15 * r18;
    r41 = fmaf(r10, r41, r40);
    r42 = r10 * r25;
    r42 = r42 * r23;
    r43 = r10 * r19;
    r43 = fmaf(r26, r43, r42);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r44,
                                         r45,
                                         r46);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r47);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r48 = 2.71828182845904523536;
    r47 = powf(r48, r47);
    r46 = r46 * r47;
    r48 = r25 * r25;
    r48 = r48 * r27;
    r49 = r37 + r48;
    r50 = r19 * r19;
    r50 = r50 * r27;
    r49 = r49 + r50;
    r35 = fmaf(r32, r39, r35);
    r35 = fmaf(r31, r41, r35);
    r35 = fmaf(r8, r43, r35);
    r35 = r35 + r46;
    r35 = fmaf(r9, r49, r35);
    r51 = copysign(1.0, r35);
    r51 = fmaf(r2, r51, r35);
    r2 = 1.0 / r51;
    r35 = r23 * r23;
    r35 = r35 * r27;
    r52 = r37 + r35;
    r52 = r52 + r48;
    r48 = r10 * r25;
    r48 = r48 * r19;
    r53 = fmaf(r23, r28, r48);
    r54 = fmaf(r8, r53, r7 * r52);
    r55 = r10 * r25;
    r55 = fmaf(r26, r55, r20);
    r20 = r16 * r18;
    r20 = fmaf(r10, r20, r33);
    r33 = r15 * r16;
    r33 = r33 * r10;
    r56 = r17 * r18;
    r56 = fmaf(r27, r56, r33);
    r57 = r17 * r17;
    r57 = r27 * r57;
    r38 = r57 + r38;
    r54 = fmaf(r9, r55, r54);
    r54 = fmaf(r32, r20, r54);
    r54 = fmaf(r31, r56, r54);
    r54 = fmaf(r30, r38, r54);
    r54 = fmaf(r44, r47, r54);
    r54 = r0 * r54;
    r4 = fmaf(r2, r54, r4);
  };
  LoadUnique<1, float, float>(reprojection_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r58);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r58 = r58 * r58;
    r58 = 1.0 / r58;
    r5 = fmaf(r5, r6, r3);
    r3 = r10 * r23;
    r3 = fmaf(r26, r3, r48);
    r48 = r17 * r18;
    r48 = fmaf(r10, r48, r33);
    r30 = fmaf(r30, r48, r7 * r3);
    r57 = r37 + r57;
    r57 = r57 + r36;
    r36 = r15 * r18;
    r36 = fmaf(r27, r36, r40);
    r42 = fmaf(r19, r28, r42);
    r35 = r37 + r35;
    r35 = r35 + r50;
    r30 = fmaf(r31, r57, r30);
    r30 = fmaf(r32, r36, r30);
    r30 = fmaf(r9, r42, r30);
    r30 = fmaf(r45, r47, r30);
    r30 = fmaf(r8, r35, r30);
    r30 = r1 * r30;
    r5 = fmaf(r2, r30, r5);
    r32 = fmaf(r4, r4, r5 * r5);
    r32 = fmaf(r32, r58, r37);
    r31 = sqrtf(r32);
    r31 = r37 + r31;
    r37 = 1.0 / r31;
    r50 = r10 * r37;
    r40 = sqrtf(r50);
    r33 = r4 * r40;
    r59 = r5 * r40;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r33, r59);
    r59 = r5 * r37;
    r60 = r10 * r5;
    r61 = r10 * r4;
    r61 = r61 * r4;
    r61 = fmaf(r37, r61, r60 * r59);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r61);
  if (global_thread_idx < problem_size) {
    r61 = r10 * r26;
    r59 = 5.00000000000000000e-01;
    r62 = fmaf(r59, r22, r59 * r21);
    r63 = -5.00000000000000000e-01;
    r64 = r14 * r59;
    r62 = fmaf(r63, r24, r62);
    r62 = fmaf(r17, r64, r62);
    r65 = r11 * r18;
    r66 = r14 * r15;
    r66 = fmaf(r63, r66, r63 * r65);
    r65 = r13 * r16;
    r66 = fmaf(r63, r65, r66);
    r67 = r12 * r17;
    r66 = fmaf(r59, r67, r66);
    r67 = r25 * r66;
    r61 = fmaf(r10, r67, r62 * r61);
    r65 = r10 * r19;
    r68 = r12 * r18;
    r69 = r13 * r15;
    r69 = fmaf(r59, r69, r63 * r68);
    r68 = r14 * r16;
    r69 = fmaf(r63, r68, r69);
    r70 = r11 * r17;
    r69 = fmaf(r63, r70, r69);
    r70 = r10 * r23;
    r68 = r11 * r15;
    r71 = r12 * r16;
    r71 = fmaf(r63, r71, r63 * r68);
    r68 = r13 * r17;
    r71 = fmaf(r63, r68, r71);
    r71 = fmaf(r18, r64, r71);
    r70 = r70 * r71;
    r65 = fmaf(r69, r65, r70);
    r61 = r61 + r65;
    r68 = r10 * r25;
    r68 = r68 * r71;
    r72 = r10 * r19;
    r72 = r72 * r62;
    r73 = r68 + r72;
    r74 = r23 * r27;
    r73 = fmaf(r66, r74, r73);
    r73 = fmaf(r69, r28, r73);
    r73 = fmaf(r8, r73, r9 * r61);
    r61 = r25 * r62;
    r74 = -4.00000000000000000e+00;
    r61 = r61 * r74;
    r75 = r23 * r69;
    r76 = r74 * r75;
    r77 = r61 + r76;
    r73 = fmaf(r7, r77, r73);
    r77 = r0 * r73;
    r78 = r10 * r25;
    r78 = r78 * r69;
    r79 = r10 * r23;
    r79 = fmaf(r62, r79, r78);
    r80 = r10 * r19;
    r80 = r80 * r66;
    r81 = r10 * r26;
    r81 = r81 * r71;
    r82 = r80 + r81;
    r83 = r79 + r82;
    r62 = fmaf(r62, r28, r27 * r67);
    r62 = r62 + r65;
    r62 = fmaf(r7, r62, r8 * r83);
    r83 = r19 * r74;
    r84 = r71 * r83;
    r61 = r61 + r84;
    r62 = fmaf(r9, r61, r62);
    r51 = r51 * r51;
    r51 = 1.0 / r51;
    r51 = r6 * r51;
    r54 = r51 * r54;
    r77 = fmaf(r62, r54, r2 * r77);
    r61 = r10 * r4;
    r85 = r62 * r51;
    r86 = r19 * r27;
    r87 = r71 * r28;
    r86 = fmaf(r66, r86, r87);
    r86 = r86 + r79;
    r84 = r76 + r84;
    r84 = fmaf(r8, r84, r9 * r86);
    r86 = r10 * r26;
    r86 = fmaf(r69, r86, r72);
    r72 = r10 * r23;
    r72 = fmaf(r66, r72, r68);
    r86 = r86 + r72;
    r84 = fmaf(r7, r86, r84);
    r86 = r1 * r84;
    r86 = fmaf(r2, r86, r30 * r85);
    r61 = fmaf(r86, r60, r77 * r61);
    r58 = r63 * r58;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r50 = rsqrtf(r50);
    r32 = rsqrtf(r32);
    r58 = r58 * r31;
    r58 = r58 * r50;
    r58 = r58 * r32;
    r61 = r61 * r58;
    r77 = fmaf(r4, r61, r77 * r40);
    r86 = fmaf(r86, r40, r5 * r61);
    r61 = r10 * r4;
    r32 = r25 * r27;
    r32 = fmaf(r69, r32, r87);
    r50 = r10 * r23;
    r31 = r14 * r17;
    r21 = fmaf(r63, r21, r63 * r31);
    r21 = fmaf(r63, r22, r21);
    r21 = fmaf(r59, r24, r21);
    r50 = r50 * r21;
    r24 = r10 * r19;
    r22 = r11 * r18;
    r31 = r13 * r16;
    r31 = fmaf(r59, r31, r59 * r22);
    r22 = r12 * r17;
    r31 = fmaf(r63, r22, r31);
    r31 = fmaf(r15, r64, r31);
    r24 = fmaf(r31, r24, r50);
    r32 = r32 + r24;
    r22 = r10 * r25;
    r22 = r22 * r31;
    r85 = r10 * r26;
    r85 = fmaf(r21, r85, r22);
    r85 = r85 + r65;
    r85 = fmaf(r8, r85, r7 * r32);
    r32 = r25 * r71;
    r32 = r32 * r74;
    r65 = r21 * r83;
    r68 = r32 + r65;
    r85 = fmaf(r9, r68, r85);
    r81 = r78 + r81;
    r81 = r81 + r24;
    r24 = r23 * r74;
    r24 = r24 * r31;
    r32 = r24 + r32;
    r32 = fmaf(r7, r32, r9 * r81);
    r81 = fmaf(r31, r28, r27 * r75);
    r78 = r10 * r19;
    r78 = r78 * r71;
    r68 = r10 * r25;
    r68 = fmaf(r21, r68, r78);
    r81 = r81 + r68;
    r32 = fmaf(r8, r81, r32);
    r81 = r0 * r32;
    r81 = fmaf(r2, r81, r85 * r54);
    r76 = r19 * r27;
    r76 = fmaf(r69, r76, r70);
    r76 = r76 + r22;
    r76 = fmaf(r21, r28, r76);
    r22 = r10 * r26;
    r75 = fmaf(r10, r75, r31 * r22);
    r75 = r75 + r68;
    r75 = fmaf(r7, r75, r9 * r76);
    r65 = r24 + r65;
    r75 = fmaf(r8, r65, r75);
    r65 = r1 * r75;
    r24 = r85 * r51;
    r24 = fmaf(r30, r24, r2 * r65);
    r61 = fmaf(r24, r60, r81 * r61);
    r65 = r4 * r61;
    r81 = fmaf(r81, r40, r58 * r65);
    r65 = r5 * r61;
    r24 = fmaf(r24, r40, r58 * r65);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r77,
                                          r86,
                                          r81,
                                          r24);
    r65 = r10 * r4;
    r76 = r10 * r25;
    r22 = r12 * r18;
    r31 = r13 * r15;
    r31 = fmaf(r63, r31, r59 * r22);
    r22 = r11 * r17;
    r31 = fmaf(r59, r22, r31);
    r31 = fmaf(r16, r64, r31);
    r76 = r76 * r31;
    r80 = r80 + r76;
    r64 = r23 * r27;
    r80 = fmaf(r21, r64, r80);
    r80 = r80 + r87;
    r71 = r23 * r71;
    r71 = r71 * r74;
    r67 = r74 * r67;
    r74 = r71 + r67;
    r74 = fmaf(r7, r74, r8 * r80);
    r80 = r10 * r23;
    r80 = r80 * r31;
    r87 = r10 * r26;
    r87 = fmaf(r66, r87, r80);
    r87 = r87 + r68;
    r74 = fmaf(r9, r87, r74);
    r87 = r0 * r74;
    r83 = r31 * r83;
    r67 = r67 + r83;
    r80 = r78 + r80;
    r78 = r25 * r27;
    r80 = fmaf(r21, r78, r80);
    r80 = fmaf(r66, r28, r80);
    r80 = fmaf(r7, r80, r9 * r67);
    r67 = r10 * r19;
    r66 = r10 * r26;
    r66 = fmaf(r31, r66, r21 * r67);
    r66 = r66 + r72;
    r80 = fmaf(r8, r66, r80);
    r87 = fmaf(r80, r54, r2 * r87);
    r66 = r80 * r51;
    r76 = r50 + r76;
    r76 = r76 + r82;
    r82 = r19 * r27;
    r28 = fmaf(r31, r28, r21 * r82);
    r28 = r28 + r72;
    r28 = fmaf(r9, r28, r7 * r76);
    r83 = r71 + r83;
    r28 = fmaf(r8, r83, r28);
    r83 = r1 * r28;
    r83 = fmaf(r2, r83, r30 * r66);
    r65 = fmaf(r83, r60, r87 * r65);
    r66 = r4 * r65;
    r87 = fmaf(r87, r40, r58 * r66);
    r66 = r5 * r65;
    r83 = fmaf(r83, r40, r58 * r66);
    r66 = r0 * r38;
    r66 = fmaf(r2, r66, r34 * r54);
    r8 = r10 * r4;
    r71 = r34 * r51;
    r9 = r1 * r48;
    r9 = fmaf(r2, r9, r30 * r71);
    r8 = fmaf(r9, r60, r66 * r8);
    r71 = r4 * r8;
    r71 = fmaf(r58, r71, r66 * r40);
    r66 = r5 * r8;
    r66 = fmaf(r58, r66, r9 * r40);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r87,
                                          r83,
                                          r71,
                                          r66);
    r9 = r0 * r56;
    r9 = fmaf(r2, r9, r41 * r54);
    r76 = r10 * r4;
    r7 = r1 * r57;
    r72 = r41 * r51;
    r72 = fmaf(r30, r72, r2 * r7);
    r76 = fmaf(r72, r60, r9 * r76);
    r7 = r4 * r76;
    r7 = fmaf(r58, r7, r9 * r40);
    r9 = r5 * r76;
    r72 = fmaf(r72, r40, r58 * r9);
    r9 = r10 * r4;
    r31 = r0 * r20;
    r31 = fmaf(r2, r31, r39 * r54);
    r82 = r1 * r36;
    r21 = r39 * r51;
    r21 = fmaf(r30, r21, r2 * r82);
    r9 = fmaf(r21, r60, r31 * r9);
    r82 = r4 * r9;
    r31 = fmaf(r31, r40, r58 * r82);
    r82 = r5 * r9;
    r21 = fmaf(r21, r40, r58 * r82);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r7,
                                          r72,
                                          r31,
                                          r21);
    r33 = r6 * r33;
    r82 = r6 * r5;
    r82 = r82 * r86;
    r82 = fmaf(r40, r82, r77 * r33);
    r50 = r6 * r5;
    r50 = r50 * r24;
    r50 = fmaf(r81, r33, r40 * r50);
    r67 = r6 * r5;
    r67 = r67 * r83;
    r67 = fmaf(r87, r33, r40 * r67);
    r78 = r6 * r5;
    r78 = r78 * r66;
    r78 = fmaf(r40, r78, r71 * r33);
    WriteSum4<float, float>((float*)inout_shared, r82, r50, r67, r78);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r6 * r5;
    r78 = r78 * r72;
    r78 = fmaf(r7, r33, r40 * r78);
    r67 = r6 * r5;
    r67 = r67 * r21;
    r67 = fmaf(r31, r33, r40 * r67);
    WriteSum2<float, float>((float*)inout_shared, r78, r67);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fmaf(r86, r86, r77 * r77);
    r78 = fmaf(r81, r81, r24 * r24);
    r50 = fmaf(r87, r87, r83 * r83);
    r82 = fmaf(r66, r66, r71 * r71);
    WriteSum4<float, float>((float*)inout_shared, r67, r78, r50, r82);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fmaf(r7, r7, r72 * r72);
    r50 = fmaf(r31, r31, r21 * r21);
    WriteSum2<float, float>((float*)inout_shared, r82, r50);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fmaf(r86, r24, r77 * r81);
    r82 = fmaf(r77, r87, r86 * r83);
    r78 = fmaf(r86, r66, r77 * r71);
    r67 = fmaf(r86, r72, r77 * r7);
    WriteSum4<float, float>((float*)inout_shared, r50, r82, r78, r67);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r77, r31, r86 * r21);
    r86 = fmaf(r24, r83, r81 * r87);
    r67 = fmaf(r24, r66, r81 * r71);
    r78 = fmaf(r81, r7, r24 * r72);
    WriteSum4<float, float>((float*)inout_shared, r77, r86, r67, r78);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r24, r21, r81 * r31);
    r81 = fmaf(r87, r71, r83 * r66);
    r78 = fmaf(r83, r72, r87 * r7);
    r87 = fmaf(r87, r31, r83 * r21);
    WriteSum4<float, float>((float*)inout_shared, r24, r81, r78, r87);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = fmaf(r66, r72, r71 * r7);
    r66 = fmaf(r66, r21, r71 * r31);
    r21 = fmaf(r72, r21, r7 * r31);
    WriteSum3<float, float>((float*)inout_shared, r87, r66, r21);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r44 * r0;
    r21 = r21 * r47;
    r21 = fmaf(r2, r21, r54 * r46);
    r66 = r10 * r4;
    r87 = r45 * r1;
    r87 = r87 * r47;
    r47 = r51 * r30;
    r47 = fmaf(r46, r47, r2 * r87);
    r66 = fmaf(r47, r60, r21 * r66);
    r87 = r4 * r66;
    r87 = fmaf(r58, r87, r21 * r40);
    r21 = r5 * r66;
    r21 = fmaf(r58, r21, r47 * r40);
    WriteIdx2<1024, float, float, float2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r87,
        r21);
    r47 = r6 * r5;
    r47 = r47 * r21;
    r47 = fmaf(r87, r33, r40 * r47);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r47);
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r21, r21, r87 * r87);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r21);
  if (global_thread_idx < problem_size) {
    r21 = r0 * r52;
    r21 = fmaf(r29, r54, r2 * r21);
    r87 = r10 * r4;
    r47 = r1 * r3;
    r46 = r29 * r51;
    r46 = fmaf(r30, r46, r2 * r47);
    r87 = fmaf(r46, r60, r21 * r87);
    r47 = r4 * r87;
    r47 = fmaf(r58, r47, r21 * r40);
    r21 = r5 * r87;
    r21 = fmaf(r58, r21, r46 * r40);
    r46 = r10 * r4;
    r72 = r0 * r53;
    r72 = fmaf(r43, r54, r2 * r72);
    r31 = r43 * r51;
    r7 = r1 * r35;
    r7 = fmaf(r2, r7, r30 * r31);
    r46 = fmaf(r7, r60, r72 * r46);
    r31 = r4 * r46;
    r72 = fmaf(r72, r40, r58 * r31);
    r31 = r5 * r46;
    r7 = fmaf(r7, r40, r58 * r31);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r21,
                                          r72,
                                          r7);
    r31 = r10 * r4;
    r71 = r0 * r55;
    r71 = fmaf(r2, r71, r49 * r54);
    r54 = r49 * r51;
    r78 = r1 * r42;
    r78 = fmaf(r2, r78, r30 * r54);
    r60 = fmaf(r78, r60, r71 * r31);
    r31 = r4 * r60;
    r71 = fmaf(r71, r40, r58 * r31);
    r31 = r5 * r60;
    r78 = fmaf(r78, r40, r58 * r31);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r71,
                                          r78);
    r31 = r6 * r5;
    r31 = r31 * r21;
    r31 = fmaf(r47, r33, r40 * r31);
    r58 = r6 * r5;
    r58 = r58 * r7;
    r58 = fmaf(r72, r33, r40 * r58);
    r54 = r6 * r5;
    r54 = r54 * r78;
    r54 = fmaf(r40, r54, r71 * r33);
    WriteSum3<float, float>((float*)inout_shared, r31, r58, r54);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r47, r47, r21 * r21);
    r58 = fmaf(r7, r7, r72 * r72);
    r31 = fmaf(r78, r78, r71 * r71);
    WriteSum3<float, float>((float*)inout_shared, r54, r58, r31);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r47, r72, r21 * r7);
    r21 = fmaf(r21, r78, r47 * r71);
    r71 = fmaf(r72, r71, r7 * r78);
    WriteSum3<float, float>((float*)inout_shared, r31, r21, r71);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                       out_sensor_from_rig_log_scale_precond_diag,
                       1);
}

void FixedRigPinholeResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* calib,
    unsigned int calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    const float* const reprojection_loss_scale,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    float* const out_sensor_from_rig_log_scale_njtr,
    float* const out_sensor_from_rig_log_scale_precond_diag,
    float* const out_sensor_from_rig_log_scale_precond_tril,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
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
      reprojection_loss_scale,
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