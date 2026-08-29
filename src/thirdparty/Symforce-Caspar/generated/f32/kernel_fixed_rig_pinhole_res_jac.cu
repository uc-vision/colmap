#include "kernel_fixed_rig_pinhole_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedRigPinholeResJacKernel(
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

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_precond_diag_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86;

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
    r37 = r10 * r37;
    r50 = sqrtf(r37);
    r40 = r4 * r50;
    r33 = r5 * r50;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r40, r33);
    r33 = r10 * r26;
    r59 = 5.00000000000000000e-01;
    r60 = fmaf(r59, r22, r59 * r21);
    r61 = -5.00000000000000000e-01;
    r62 = r14 * r59;
    r60 = fmaf(r61, r24, r60);
    r60 = fmaf(r17, r62, r60);
    r63 = r11 * r18;
    r64 = r14 * r15;
    r64 = fmaf(r61, r64, r61 * r63);
    r63 = r13 * r16;
    r64 = fmaf(r61, r63, r64);
    r65 = r12 * r17;
    r64 = fmaf(r59, r65, r64);
    r65 = r25 * r64;
    r33 = fmaf(r10, r65, r60 * r33);
    r63 = r10 * r19;
    r66 = r12 * r18;
    r67 = r13 * r15;
    r67 = fmaf(r59, r67, r61 * r66);
    r66 = r14 * r16;
    r67 = fmaf(r61, r66, r67);
    r68 = r11 * r17;
    r67 = fmaf(r61, r68, r67);
    r68 = r10 * r23;
    r66 = r11 * r15;
    r69 = r12 * r16;
    r69 = fmaf(r61, r69, r61 * r66);
    r66 = r13 * r17;
    r69 = fmaf(r61, r66, r69);
    r69 = fmaf(r18, r62, r69);
    r68 = r68 * r69;
    r63 = fmaf(r67, r63, r68);
    r33 = r33 + r63;
    r66 = r10 * r25;
    r66 = r66 * r69;
    r70 = r10 * r19;
    r70 = r70 * r60;
    r71 = r66 + r70;
    r72 = r23 * r27;
    r71 = fmaf(r64, r72, r71);
    r71 = fmaf(r67, r28, r71);
    r71 = fmaf(r8, r71, r9 * r33);
    r33 = r25 * r60;
    r72 = -4.00000000000000000e+00;
    r33 = r33 * r72;
    r73 = r23 * r67;
    r74 = r72 * r73;
    r75 = r33 + r74;
    r71 = fmaf(r7, r75, r71);
    r75 = r0 * r71;
    r76 = r10 * r25;
    r76 = r76 * r67;
    r77 = r10 * r23;
    r77 = fmaf(r60, r77, r76);
    r78 = r10 * r19;
    r78 = r78 * r64;
    r79 = r10 * r26;
    r79 = r79 * r69;
    r80 = r78 + r79;
    r81 = r77 + r80;
    r60 = fmaf(r60, r28, r27 * r65);
    r60 = r60 + r63;
    r60 = fmaf(r7, r60, r8 * r81);
    r81 = r19 * r72;
    r82 = r69 * r81;
    r33 = r33 + r82;
    r60 = fmaf(r9, r33, r60);
    r51 = r51 * r51;
    r51 = 1.0 / r51;
    r51 = r6 * r51;
    r54 = r51 * r54;
    r75 = fmaf(r60, r54, r2 * r75);
    r33 = r10 * r4;
    r83 = r10 * r5;
    r84 = r60 * r51;
    r85 = r19 * r27;
    r86 = r69 * r28;
    r85 = fmaf(r64, r85, r86);
    r85 = r85 + r77;
    r82 = r74 + r82;
    r82 = fmaf(r8, r82, r9 * r85);
    r85 = r10 * r26;
    r85 = fmaf(r67, r85, r70);
    r70 = r10 * r23;
    r70 = fmaf(r64, r70, r66);
    r85 = r85 + r70;
    r82 = fmaf(r7, r85, r82);
    r85 = r1 * r82;
    r85 = fmaf(r2, r85, r30 * r84);
    r33 = fmaf(r85, r83, r75 * r33);
    r58 = r61 * r58;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r37 = rsqrtf(r37);
    r32 = rsqrtf(r32);
    r58 = r58 * r31;
    r58 = r58 * r37;
    r58 = r58 * r32;
    r33 = r33 * r58;
    r75 = fmaf(r4, r33, r75 * r50);
    r85 = fmaf(r85, r50, r5 * r33);
    r33 = r10 * r4;
    r32 = r25 * r27;
    r32 = fmaf(r67, r32, r86);
    r37 = r10 * r23;
    r31 = r14 * r17;
    r21 = fmaf(r61, r21, r61 * r31);
    r21 = fmaf(r61, r22, r21);
    r21 = fmaf(r59, r24, r21);
    r37 = r37 * r21;
    r24 = r10 * r19;
    r22 = r11 * r18;
    r31 = r13 * r16;
    r31 = fmaf(r59, r31, r59 * r22);
    r22 = r12 * r17;
    r31 = fmaf(r61, r22, r31);
    r31 = fmaf(r15, r62, r31);
    r24 = fmaf(r31, r24, r37);
    r32 = r32 + r24;
    r22 = r10 * r25;
    r22 = r22 * r31;
    r84 = r10 * r26;
    r84 = fmaf(r21, r84, r22);
    r84 = r84 + r63;
    r84 = fmaf(r8, r84, r7 * r32);
    r32 = r25 * r69;
    r32 = r32 * r72;
    r63 = r21 * r81;
    r66 = r32 + r63;
    r84 = fmaf(r9, r66, r84);
    r79 = r76 + r79;
    r79 = r79 + r24;
    r24 = r23 * r72;
    r24 = r24 * r31;
    r32 = r24 + r32;
    r32 = fmaf(r7, r32, r9 * r79);
    r79 = fmaf(r31, r28, r27 * r73);
    r76 = r10 * r19;
    r76 = r76 * r69;
    r66 = r10 * r25;
    r66 = fmaf(r21, r66, r76);
    r79 = r79 + r66;
    r32 = fmaf(r8, r79, r32);
    r79 = r0 * r32;
    r79 = fmaf(r2, r79, r84 * r54);
    r74 = r19 * r27;
    r74 = fmaf(r67, r74, r68);
    r74 = r74 + r22;
    r74 = fmaf(r21, r28, r74);
    r22 = r10 * r26;
    r73 = fmaf(r10, r73, r31 * r22);
    r73 = r73 + r66;
    r73 = fmaf(r7, r73, r9 * r74);
    r63 = r24 + r63;
    r73 = fmaf(r8, r63, r73);
    r63 = r1 * r73;
    r24 = r84 * r51;
    r24 = fmaf(r30, r24, r2 * r63);
    r33 = fmaf(r24, r83, r79 * r33);
    r63 = r4 * r33;
    r79 = fmaf(r79, r50, r58 * r63);
    r63 = r5 * r33;
    r24 = fmaf(r24, r50, r58 * r63);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r75,
                                          r85,
                                          r79,
                                          r24);
    r63 = r10 * r4;
    r74 = r10 * r25;
    r22 = r12 * r18;
    r31 = r13 * r15;
    r31 = fmaf(r61, r31, r59 * r22);
    r22 = r11 * r17;
    r31 = fmaf(r59, r22, r31);
    r31 = fmaf(r16, r62, r31);
    r74 = r74 * r31;
    r78 = r78 + r74;
    r62 = r23 * r27;
    r78 = fmaf(r21, r62, r78);
    r78 = r78 + r86;
    r69 = r23 * r69;
    r69 = r69 * r72;
    r65 = r72 * r65;
    r72 = r69 + r65;
    r72 = fmaf(r7, r72, r8 * r78);
    r78 = r10 * r23;
    r78 = r78 * r31;
    r86 = r10 * r26;
    r86 = fmaf(r64, r86, r78);
    r86 = r86 + r66;
    r72 = fmaf(r9, r86, r72);
    r86 = r0 * r72;
    r81 = r31 * r81;
    r65 = r65 + r81;
    r78 = r76 + r78;
    r76 = r25 * r27;
    r78 = fmaf(r21, r76, r78);
    r78 = fmaf(r64, r28, r78);
    r78 = fmaf(r7, r78, r9 * r65);
    r65 = r10 * r19;
    r64 = r10 * r26;
    r64 = fmaf(r31, r64, r21 * r65);
    r64 = r64 + r70;
    r78 = fmaf(r8, r64, r78);
    r86 = fmaf(r78, r54, r2 * r86);
    r64 = r78 * r51;
    r74 = r37 + r74;
    r74 = r74 + r80;
    r80 = r19 * r27;
    r28 = fmaf(r31, r28, r21 * r80);
    r28 = r28 + r70;
    r28 = fmaf(r9, r28, r7 * r74);
    r81 = r69 + r81;
    r28 = fmaf(r8, r81, r28);
    r81 = r1 * r28;
    r81 = fmaf(r2, r81, r30 * r64);
    r63 = fmaf(r81, r83, r86 * r63);
    r64 = r4 * r63;
    r86 = fmaf(r86, r50, r58 * r64);
    r64 = r5 * r63;
    r81 = fmaf(r81, r50, r58 * r64);
    r64 = r0 * r38;
    r64 = fmaf(r2, r64, r34 * r54);
    r8 = r10 * r4;
    r69 = r34 * r51;
    r9 = r1 * r48;
    r9 = fmaf(r2, r9, r30 * r69);
    r8 = fmaf(r9, r83, r64 * r8);
    r69 = r4 * r8;
    r69 = fmaf(r58, r69, r64 * r50);
    r64 = r5 * r8;
    r64 = fmaf(r58, r64, r9 * r50);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r86,
                                          r81,
                                          r69,
                                          r64);
    r9 = r0 * r56;
    r9 = fmaf(r2, r9, r41 * r54);
    r74 = r10 * r4;
    r7 = r1 * r57;
    r70 = r41 * r51;
    r70 = fmaf(r30, r70, r2 * r7);
    r74 = fmaf(r70, r83, r9 * r74);
    r7 = r4 * r74;
    r7 = fmaf(r58, r7, r9 * r50);
    r9 = r5 * r74;
    r70 = fmaf(r70, r50, r58 * r9);
    r9 = r10 * r4;
    r31 = r0 * r20;
    r31 = fmaf(r2, r31, r39 * r54);
    r80 = r1 * r36;
    r21 = r39 * r51;
    r21 = fmaf(r30, r21, r2 * r80);
    r9 = fmaf(r21, r83, r31 * r9);
    r80 = r4 * r9;
    r31 = fmaf(r31, r50, r58 * r80);
    r80 = r5 * r9;
    r21 = fmaf(r21, r50, r58 * r80);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r7,
                                          r70,
                                          r31,
                                          r21);
    r40 = r6 * r40;
    r80 = r6 * r5;
    r80 = r80 * r85;
    r80 = fmaf(r50, r80, r75 * r40);
    r37 = r6 * r5;
    r37 = r37 * r24;
    r37 = fmaf(r79, r40, r50 * r37);
    r65 = r6 * r5;
    r65 = r65 * r81;
    r65 = fmaf(r86, r40, r50 * r65);
    r76 = r6 * r5;
    r76 = r76 * r64;
    r76 = fmaf(r50, r76, r69 * r40);
    WriteSum4<float, float>((float*)inout_shared, r80, r37, r65, r76);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r6 * r5;
    r76 = r76 * r70;
    r76 = fmaf(r7, r40, r50 * r76);
    r65 = r6 * r5;
    r65 = r65 * r21;
    r65 = fmaf(r31, r40, r50 * r65);
    WriteSum2<float, float>((float*)inout_shared, r76, r65);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fmaf(r85, r85, r75 * r75);
    r76 = fmaf(r79, r79, r24 * r24);
    r37 = fmaf(r86, r86, r81 * r81);
    r80 = fmaf(r64, r64, r69 * r69);
    WriteSum4<float, float>((float*)inout_shared, r65, r76, r37, r80);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fmaf(r7, r7, r70 * r70);
    r37 = fmaf(r31, r31, r21 * r21);
    WriteSum2<float, float>((float*)inout_shared, r80, r37);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fmaf(r85, r24, r75 * r79);
    r80 = fmaf(r75, r86, r85 * r81);
    r76 = fmaf(r85, r64, r75 * r69);
    r65 = fmaf(r85, r70, r75 * r7);
    WriteSum4<float, float>((float*)inout_shared, r37, r80, r76, r65);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r75, r31, r85 * r21);
    r85 = fmaf(r24, r81, r79 * r86);
    r65 = fmaf(r24, r64, r79 * r69);
    r76 = fmaf(r79, r7, r24 * r70);
    WriteSum4<float, float>((float*)inout_shared, r75, r85, r65, r76);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r24, r21, r79 * r31);
    r79 = fmaf(r86, r69, r81 * r64);
    r76 = fmaf(r81, r70, r86 * r7);
    r86 = fmaf(r86, r31, r81 * r21);
    WriteSum4<float, float>((float*)inout_shared, r24, r79, r76, r86);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fmaf(r64, r70, r69 * r7);
    r64 = fmaf(r64, r21, r69 * r31);
    r21 = fmaf(r70, r21, r7 * r31);
    WriteSum3<float, float>((float*)inout_shared, r86, r64, r21);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r44 * r0;
    r21 = r21 * r47;
    r21 = fmaf(r2, r21, r54 * r46);
    r64 = r10 * r4;
    r86 = r45 * r1;
    r86 = r86 * r47;
    r47 = r51 * r30;
    r47 = fmaf(r46, r47, r2 * r86);
    r64 = fmaf(r47, r83, r21 * r64);
    r86 = r4 * r64;
    r86 = fmaf(r58, r86, r21 * r50);
    r21 = r5 * r64;
    r21 = fmaf(r58, r21, r47 * r50);
    WriteIdx2<1024, float, float, float2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r86,
        r21);
    r47 = r6 * r5;
    r47 = r47 * r21;
    r47 = fmaf(r86, r40, r50 * r47);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r47);
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r21, r21, r86 * r86);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r21);
  if (global_thread_idx < problem_size) {
    r21 = r0 * r52;
    r21 = fmaf(r29, r54, r2 * r21);
    r86 = r10 * r4;
    r47 = r1 * r3;
    r46 = r29 * r51;
    r46 = fmaf(r30, r46, r2 * r47);
    r86 = fmaf(r46, r83, r21 * r86);
    r47 = r4 * r86;
    r47 = fmaf(r58, r47, r21 * r50);
    r21 = r5 * r86;
    r21 = fmaf(r58, r21, r46 * r50);
    r46 = r10 * r4;
    r70 = r0 * r53;
    r70 = fmaf(r43, r54, r2 * r70);
    r31 = r43 * r51;
    r7 = r1 * r35;
    r7 = fmaf(r2, r7, r30 * r31);
    r46 = fmaf(r7, r83, r70 * r46);
    r31 = r4 * r46;
    r70 = fmaf(r70, r50, r58 * r31);
    r31 = r5 * r46;
    r7 = fmaf(r7, r50, r58 * r31);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r21,
                                          r70,
                                          r7);
    r31 = r10 * r4;
    r69 = r0 * r55;
    r69 = fmaf(r2, r69, r49 * r54);
    r54 = r49 * r51;
    r76 = r1 * r42;
    r76 = fmaf(r2, r76, r30 * r54);
    r83 = fmaf(r76, r83, r69 * r31);
    r31 = r4 * r83;
    r69 = fmaf(r69, r50, r58 * r31);
    r31 = r5 * r83;
    r76 = fmaf(r76, r50, r58 * r31);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r69,
                                          r76);
    r31 = r6 * r5;
    r31 = r31 * r21;
    r31 = fmaf(r47, r40, r50 * r31);
    r58 = r6 * r5;
    r58 = r58 * r7;
    r58 = fmaf(r70, r40, r50 * r58);
    r54 = r6 * r5;
    r54 = r54 * r76;
    r54 = fmaf(r50, r54, r69 * r40);
    WriteSum3<float, float>((float*)inout_shared, r31, r58, r54);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r47, r47, r21 * r21);
    r58 = fmaf(r7, r7, r70 * r70);
    r31 = fmaf(r76, r76, r69 * r69);
    WriteSum3<float, float>((float*)inout_shared, r54, r58, r31);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r47, r70, r21 * r7);
    r21 = fmaf(r21, r76, r47 * r69);
    r69 = fmaf(r70, r69, r7 * r76);
    WriteSum3<float, float>((float*)inout_shared, r31, r21, r69);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                       out_sensor_from_rig_log_scale_precond_diag,
                       1);
}

void FixedRigPinholeResJac(
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
  FixedRigPinholeResJacKernel<<<n_blocks, 1024>>>(
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