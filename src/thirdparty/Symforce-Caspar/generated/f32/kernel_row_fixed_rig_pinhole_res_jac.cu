#include "kernel_row_fixed_rig_pinhole_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) RowFixedRigPinholeResJacKernel(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_calibration,
    unsigned int sensor_calibration_num_alloc,
    SharedIndex* sensor_calibration_indices,
    const float* const sensor_from_rig_log_scale,
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
  __shared__ SharedIndex sensor_calibration_indices_loc[1024];
  sensor_calibration_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? sensor_calibration_indices[global_thread_idx]
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85;
  LoadShared<3, float, float>(sensor_calibration,
                              8 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r1);
    r1 = 9.99999999999999955e-07;
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r6,
                       r7,
                       r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = 2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  LoadShared<4, float, float>(sensor_calibration,
                              0 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r14,
                       r15,
                       r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r12 * r17;
    r19 = r11 * r14;
    r20 = r18 + r19;
    r21 = r10 * r15;
    r20 = fmaf(r13, r16, r20);
    r20 = fmaf(r5, r21, r20);
    r22 = r9 * r20;
    r23 = fmaf(r13, r14, r10 * r17);
    r24 = r11 * r16;
    r23 = fmaf(r5, r24, r23);
    r23 = fmaf(r12, r15, r23);
    r22 = r22 * r23;
    r24 = fmaf(r11, r15, r10 * r14);
    r24 = fmaf(r12, r16, r24);
    r24 = fmaf(r13, r17, r5 * r24);
    r25 = -2.00000000000000000e+00;
    r26 = r12 * r14;
    r26 = fmaf(r5, r26, r11 * r17);
    r26 = fmaf(r13, r15, r26);
    r26 = fmaf(r10, r16, r26);
    r27 = r25 * r26;
    r28 = fmaf(r24, r27, r22);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r29,
                       r30,
                       r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r14 * r14;
    r32 = r32 * r25;
    r33 = 1.00000000000000000e+00;
    r34 = r15 * r15;
    r34 = fmaf(r25, r34, r33);
    r35 = r32 + r34;
    r36 = fmaf(r31, r35, r6 * r28);
    r37 = r15 * r17;
    r38 = r14 * r16;
    r38 = r38 * r9;
    r37 = fmaf(r25, r37, r38);
    r39 = r14 * r17;
    r40 = r15 * r16;
    r40 = r40 * r9;
    r39 = fmaf(r9, r39, r40);
  };
  LoadShared<4, float, float>(sensor_calibration,
                              4 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r41,
                       r42,
                       r43,
                       r44);
  };
  __syncthreads();
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r45);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r46 = 2.71828182845904523536;
    r45 = powf(r46, r45);
    r43 = r43 * r45;
    r46 = r9 * r20;
    r46 = r46 * r26;
    r47 = r9 * r24;
    r48 = fmaf(r23, r47, r46);
    r49 = r26 * r27;
    r50 = r33 + r49;
    r51 = r25 * r23;
    r51 = r51 * r23;
    r50 = r50 + r51;
    r36 = fmaf(r29, r37, r36);
    r36 = fmaf(r30, r39, r36);
    r36 = r36 + r43;
    r36 = fmaf(r7, r48, r36);
    r36 = fmaf(r8, r50, r36);
    r52 = copysign(1.0, r36);
    r52 = fmaf(r1, r52, r36);
    r1 = 1.0 / r52;
    r22 = fmaf(r26, r47, r22);
    r36 = r16 * r16;
    r36 = r25 * r36;
    r34 = r36 + r34;
    r53 = fmaf(r29, r34, r8 * r22);
    r54 = r15 * r17;
    r54 = fmaf(r9, r54, r38);
    r38 = r14 * r15;
    r38 = r38 * r9;
    r55 = r16 * r17;
    r55 = fmaf(r25, r55, r38);
    r49 = r33 + r49;
    r56 = r25 * r20;
    r56 = r56 * r20;
    r49 = r49 + r56;
    r57 = r9 * r23;
    r57 = r57 * r26;
    r58 = r25 * r20;
    r58 = fmaf(r24, r58, r57);
    r53 = fmaf(r31, r54, r53);
    r53 = fmaf(r30, r55, r53);
    r53 = fmaf(r41, r45, r53);
    r53 = fmaf(r6, r49, r53);
    r53 = fmaf(r7, r58, r53);
    r53 = r44 * r53;
    r3 = fmaf(r1, r53, r3);
  };
  LoadUnique<1, float, float>(reprojection_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r59);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r59 = r59 * r59;
    r59 = 1.0 / r59;
    r4 = fmaf(r4, r5, r2);
    r56 = r33 + r56;
    r56 = r56 + r51;
    r36 = r33 + r36;
    r36 = r36 + r32;
    r30 = fmaf(r30, r36, r7 * r56);
    r32 = r16 * r17;
    r32 = fmaf(r9, r32, r38);
    r38 = r14 * r17;
    r38 = fmaf(r25, r38, r40);
    r40 = r25 * r23;
    r40 = fmaf(r24, r40, r46);
    r57 = fmaf(r20, r47, r57);
    r30 = fmaf(r29, r32, r30);
    r30 = fmaf(r31, r38, r30);
    r30 = fmaf(r42, r45, r30);
    r30 = fmaf(r8, r40, r30);
    r30 = fmaf(r6, r57, r30);
    r30 = r0 * r30;
    r4 = fmaf(r1, r30, r4);
    r31 = fmaf(r3, r3, r4 * r4);
    r31 = fmaf(r31, r59, r33);
    r29 = sqrtf(r31);
    r29 = r33 + r29;
    r33 = 1.0 / r29;
    r33 = r9 * r33;
    r46 = sqrtf(r33);
    r51 = r3 * r46;
    r2 = r4 * r46;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r51, r2);
    r2 = r9 * r4;
    r60 = -4.00000000000000000e+00;
    r61 = r23 * r60;
    r62 = r10 * r14;
    r63 = -5.00000000000000000e-01;
    r64 = r11 * r15;
    r64 = fmaf(r63, r64, r63 * r62);
    r62 = r12 * r16;
    r64 = fmaf(r63, r62, r64);
    r65 = 5.00000000000000000e-01;
    r66 = r13 * r65;
    r64 = fmaf(r17, r66, r64);
    r61 = r61 * r64;
    r62 = r11 * r17;
    r67 = r12 * r14;
    r67 = fmaf(r65, r67, r63 * r62);
    r62 = r13 * r15;
    r67 = fmaf(r63, r62, r67);
    r68 = r10 * r16;
    r67 = fmaf(r63, r68, r67);
    r68 = r20 * r60;
    r62 = r67 * r68;
    r69 = r61 + r62;
    r70 = r9 * r23;
    r71 = fmaf(r65, r19, r65 * r18);
    r71 = fmaf(r63, r21, r71);
    r71 = fmaf(r16, r66, r71);
    r70 = r70 * r71;
    r72 = fmaf(r67, r47, r70);
    r73 = r9 * r26;
    r73 = r73 * r64;
    r74 = r9 * r20;
    r75 = r10 * r17;
    r76 = r13 * r14;
    r76 = fmaf(r63, r76, r63 * r75);
    r75 = r12 * r15;
    r76 = fmaf(r63, r75, r76);
    r77 = r11 * r16;
    r76 = fmaf(r65, r77, r76);
    r74 = fmaf(r76, r74, r73);
    r72 = r72 + r74;
    r72 = fmaf(r6, r72, r7 * r69);
    r69 = r25 * r23;
    r77 = r25 * r24;
    r77 = r77 * r64;
    r69 = fmaf(r76, r69, r77);
    r75 = r9 * r26;
    r75 = r75 * r67;
    r78 = r9 * r20;
    r78 = fmaf(r71, r78, r75);
    r69 = r69 + r78;
    r72 = fmaf(r8, r69, r72);
    r69 = r0 * r72;
    r79 = r25 * r24;
    r79 = fmaf(r76, r27, r71 * r79);
    r80 = r9 * r20;
    r80 = r80 * r64;
    r81 = r9 * r23;
    r81 = fmaf(r67, r81, r80);
    r79 = r79 + r81;
    r82 = r9 * r23;
    r82 = r82 * r76;
    r83 = r64 * r47;
    r84 = r82 + r83;
    r78 = r78 + r84;
    r78 = fmaf(r7, r78, r6 * r79);
    r79 = r26 * r60;
    r79 = r79 * r71;
    r61 = r61 + r79;
    r78 = fmaf(r8, r61, r78);
    r52 = r52 * r52;
    r52 = 1.0 / r52;
    r52 = r5 * r52;
    r61 = r78 * r52;
    r61 = fmaf(r30, r61, r1 * r69);
    r69 = r9 * r3;
    r62 = r79 + r62;
    r70 = r73 + r70;
    r73 = r25 * r20;
    r70 = fmaf(r76, r73, r70);
    r79 = r25 * r24;
    r70 = fmaf(r67, r79, r70);
    r70 = fmaf(r7, r70, r6 * r62);
    r62 = r9 * r26;
    r71 = fmaf(r71, r47, r76 * r62);
    r71 = r71 + r81;
    r70 = fmaf(r8, r71, r70);
    r71 = r44 * r70;
    r53 = r52 * r53;
    r71 = fmaf(r78, r53, r1 * r71);
    r69 = fmaf(r71, r69, r61 * r2);
    r2 = r3 * r69;
    r59 = r63 * r59;
    r33 = rsqrtf(r33);
    r29 = r29 * r29;
    r29 = 1.0 / r29;
    r31 = rsqrtf(r31);
    r59 = r59 * r33;
    r59 = r59 * r29;
    r59 = r59 * r31;
    r71 = fmaf(r71, r46, r59 * r2);
    r2 = r4 * r59;
    r61 = fmaf(r69, r2, r61 * r46);
    r31 = r26 * r60;
    r31 = r31 * r64;
    r29 = r10 * r17;
    r33 = r12 * r15;
    r33 = fmaf(r65, r33, r65 * r29);
    r29 = r11 * r16;
    r33 = fmaf(r63, r29, r33);
    r33 = fmaf(r14, r66, r33);
    r29 = r33 * r68;
    r62 = r31 + r29;
    r83 = r75 + r83;
    r75 = r9 * r20;
    r79 = r13 * r16;
    r18 = fmaf(r63, r18, r63 * r79);
    r18 = fmaf(r63, r19, r18);
    r18 = fmaf(r65, r21, r18);
    r75 = r75 * r18;
    r21 = r9 * r23;
    r21 = fmaf(r33, r21, r75);
    r83 = r83 + r21;
    r83 = fmaf(r8, r83, r6 * r62);
    r62 = r25 * r20;
    r19 = r25 * r24;
    r19 = fmaf(r33, r19, r67 * r62);
    r62 = r9 * r26;
    r79 = r9 * r23;
    r79 = r79 * r64;
    r62 = fmaf(r18, r62, r79);
    r19 = r19 + r62;
    r83 = fmaf(r7, r19, r83);
    r19 = r44 * r83;
    r73 = r9 * r26;
    r73 = r73 * r33;
    r85 = fmaf(r18, r47, r73);
    r85 = r85 + r81;
    r81 = fmaf(r67, r27, r77);
    r81 = r81 + r21;
    r81 = fmaf(r6, r81, r7 * r85);
    r85 = r23 * r18;
    r21 = r60 * r85;
    r31 = r31 + r21;
    r81 = fmaf(r8, r31, r81);
    r19 = fmaf(r81, r53, r1 * r19);
    r31 = r9 * r4;
    r21 = r29 + r21;
    r73 = r80 + r73;
    r80 = r25 * r23;
    r73 = fmaf(r67, r80, r73);
    r29 = r25 * r24;
    r73 = fmaf(r18, r29, r73);
    r73 = fmaf(r8, r73, r7 * r21);
    r21 = r9 * r20;
    r33 = fmaf(r33, r47, r67 * r21);
    r33 = r33 + r62;
    r73 = fmaf(r6, r33, r73);
    r33 = r0 * r73;
    r21 = r81 * r52;
    r21 = fmaf(r30, r21, r1 * r33);
    r33 = r9 * r3;
    r33 = fmaf(r19, r33, r21 * r31);
    r31 = r3 * r33;
    r31 = fmaf(r59, r31, r19 * r46);
    r21 = fmaf(r33, r2, r21 * r46);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r71,
                                          r61,
                                          r31,
                                          r21);
    r19 = r11 * r17;
    r67 = r12 * r14;
    r67 = fmaf(r63, r67, r65 * r19);
    r19 = r10 * r16;
    r67 = fmaf(r65, r19, r67);
    r67 = fmaf(r15, r66, r67);
    r66 = fmaf(r67, r47, r9 * r85);
    r66 = r66 + r74;
    r19 = r9 * r20;
    r19 = r19 * r67;
    r79 = r79 + r19;
    r65 = r25 * r24;
    r79 = fmaf(r76, r65, r79);
    r79 = fmaf(r18, r27, r79);
    r79 = fmaf(r6, r79, r7 * r66);
    r66 = r26 * r60;
    r66 = r66 * r76;
    r60 = r23 * r60;
    r60 = r60 * r67;
    r27 = r66 + r60;
    r79 = fmaf(r8, r27, r79);
    r68 = r64 * r68;
    r66 = r66 + r68;
    r82 = r77 + r82;
    r77 = r9 * r26;
    r77 = r77 * r67;
    r64 = r25 * r20;
    r82 = fmaf(r18, r64, r82);
    r82 = r82 + r77;
    r82 = fmaf(r7, r82, r6 * r66);
    r47 = fmaf(r76, r47, r19);
    r47 = r47 + r62;
    r82 = fmaf(r8, r47, r82);
    r47 = r44 * r82;
    r47 = fmaf(r1, r47, r79 * r53);
    r62 = r9 * r3;
    r76 = r9 * r4;
    r19 = r79 * r52;
    r68 = r60 + r68;
    r60 = r25 * r24;
    r85 = fmaf(r25, r85, r67 * r60);
    r85 = r85 + r74;
    r85 = fmaf(r8, r85, r7 * r68);
    r77 = r75 + r77;
    r77 = r77 + r84;
    r85 = fmaf(r6, r77, r85);
    r77 = r0 * r85;
    r77 = fmaf(r1, r77, r30 * r19);
    r76 = fmaf(r77, r76, r47 * r62);
    r62 = r3 * r76;
    r62 = fmaf(r59, r62, r47 * r46);
    r77 = fmaf(r77, r46, r76 * r2);
    r47 = r9 * r3;
    r19 = r44 * r34;
    r19 = fmaf(r37, r53, r1 * r19);
    r6 = r9 * r4;
    r84 = r37 * r52;
    r75 = r0 * r32;
    r75 = fmaf(r1, r75, r30 * r84);
    r6 = fmaf(r75, r6, r19 * r47);
    r47 = r3 * r6;
    r19 = fmaf(r19, r46, r59 * r47);
    r75 = fmaf(r6, r2, r75 * r46);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r62,
                                          r77,
                                          r19,
                                          r75);
    r47 = r9 * r4;
    r84 = r0 * r36;
    r8 = r39 * r52;
    r8 = fmaf(r30, r8, r1 * r84);
    r84 = r9 * r3;
    r68 = r44 * r55;
    r68 = fmaf(r1, r68, r39 * r53);
    r84 = fmaf(r68, r84, r8 * r47);
    r47 = r3 * r84;
    r68 = fmaf(r68, r46, r59 * r47);
    r8 = fmaf(r8, r46, r84 * r2);
    r47 = r9 * r4;
    r7 = r35 * r52;
    r74 = r0 * r38;
    r74 = fmaf(r1, r74, r30 * r7);
    r7 = r9 * r3;
    r60 = r44 * r54;
    r60 = fmaf(r35, r53, r1 * r60);
    r7 = fmaf(r60, r7, r74 * r47);
    r47 = r3 * r7;
    r60 = fmaf(r60, r46, r59 * r47);
    r74 = fmaf(r74, r46, r7 * r2);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r68,
                                          r8,
                                          r60,
                                          r74);
    r47 = r5 * r4;
    r47 = r47 * r61;
    r51 = r5 * r51;
    r47 = fmaf(r71, r51, r46 * r47);
    r67 = r5 * r4;
    r67 = r67 * r21;
    r67 = fmaf(r31, r51, r46 * r67);
    r66 = r5 * r4;
    r66 = r66 * r77;
    r66 = fmaf(r62, r51, r46 * r66);
    r64 = r5 * r4;
    r64 = r64 * r75;
    r64 = fmaf(r19, r51, r46 * r64);
    WriteSum4<float, float>((float*)inout_shared, r47, r67, r66, r64);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r5 * r4;
    r64 = r64 * r8;
    r64 = fmaf(r46, r64, r68 * r51);
    r66 = r5 * r4;
    r66 = r66 * r74;
    r66 = fmaf(r46, r66, r60 * r51);
    WriteSum2<float, float>((float*)inout_shared, r64, r66);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fmaf(r61, r61, r71 * r71);
    r64 = fmaf(r21, r21, r31 * r31);
    r67 = fmaf(r77, r77, r62 * r62);
    r47 = fmaf(r75, r75, r19 * r19);
    WriteSum4<float, float>((float*)inout_shared, r66, r64, r67, r47);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r68, r68, r8 * r8);
    r67 = fmaf(r60, r60, r74 * r74);
    WriteSum2<float, float>((float*)inout_shared, r47, r67);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fmaf(r61, r21, r71 * r31);
    r47 = fmaf(r71, r62, r61 * r77);
    r64 = fmaf(r71, r19, r61 * r75);
    r66 = fmaf(r61, r8, r71 * r68);
    WriteSum4<float, float>((float*)inout_shared, r67, r47, r64, r66);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r61, r74, r71 * r60);
    r71 = fmaf(r31, r62, r21 * r77);
    r66 = fmaf(r21, r75, r31 * r19);
    r64 = fmaf(r31, r68, r21 * r8);
    WriteSum4<float, float>((float*)inout_shared, r61, r71, r66, r64);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r21, r74, r31 * r60);
    r31 = fmaf(r62, r19, r77 * r75);
    r64 = fmaf(r77, r8, r62 * r68);
    r62 = fmaf(r62, r60, r77 * r74);
    WriteSum4<float, float>((float*)inout_shared, r21, r31, r64, r62);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fmaf(r75, r8, r19 * r68);
    r75 = fmaf(r75, r74, r19 * r60);
    r60 = fmaf(r68, r60, r8 * r74);
    WriteSum3<float, float>((float*)inout_shared, r62, r75, r60);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r9 * r3;
    r75 = r41 * r44;
    r75 = r75 * r45;
    r75 = fmaf(r53, r43, r1 * r75);
    r62 = r9 * r4;
    r68 = r42 * r0;
    r68 = r68 * r45;
    r45 = r52 * r30;
    r45 = fmaf(r43, r45, r1 * r68);
    r62 = fmaf(r45, r62, r75 * r60);
    r60 = r3 * r62;
    r75 = fmaf(r75, r46, r59 * r60);
    r45 = fmaf(r62, r2, r45 * r46);
    WriteIdx2<1024, float, float, float2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r75,
        r45);
    r60 = r5 * r4;
    r60 = r60 * r45;
    r60 = fmaf(r46, r60, r75 * r51);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r60);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r75, r75, r45 * r45);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r75);
  if (global_thread_idx < problem_size) {
    r75 = r9 * r4;
    r45 = r28 * r52;
    r60 = r0 * r57;
    r60 = fmaf(r1, r60, r30 * r45);
    r45 = r9 * r3;
    r68 = r44 * r49;
    r68 = fmaf(r28, r53, r1 * r68);
    r45 = fmaf(r68, r45, r60 * r75);
    r75 = r3 * r45;
    r68 = fmaf(r68, r46, r59 * r75);
    r60 = fmaf(r45, r2, r60 * r46);
    r75 = r44 * r58;
    r75 = fmaf(r1, r75, r48 * r53);
    r43 = r9 * r3;
    r74 = r9 * r4;
    r8 = r0 * r56;
    r19 = r48 * r52;
    r19 = fmaf(r30, r19, r1 * r8);
    r74 = fmaf(r19, r74, r75 * r43);
    r43 = r3 * r74;
    r43 = fmaf(r59, r43, r75 * r46);
    r19 = fmaf(r74, r2, r19 * r46);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r68,
                                          r60,
                                          r43,
                                          r19);
    r75 = r9 * r3;
    r8 = r44 * r22;
    r8 = fmaf(r1, r8, r50 * r53);
    r53 = r9 * r4;
    r64 = r50 * r52;
    r31 = r0 * r40;
    r31 = fmaf(r1, r31, r30 * r64);
    r53 = fmaf(r31, r53, r8 * r75);
    r75 = r3 * r53;
    r8 = fmaf(r8, r46, r59 * r75);
    r2 = fmaf(r53, r2, r31 * r46);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r8, r2);
    r31 = r5 * r4;
    r31 = r31 * r60;
    r31 = fmaf(r46, r31, r68 * r51);
    r75 = r5 * r4;
    r75 = r75 * r19;
    r75 = fmaf(r46, r75, r43 * r51);
    r59 = r5 * r4;
    r59 = r59 * r2;
    r59 = fmaf(r46, r59, r8 * r51);
    WriteSum3<float, float>((float*)inout_shared, r31, r75, r59);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fmaf(r60, r60, r68 * r68);
    r75 = fmaf(r43, r43, r19 * r19);
    r31 = fmaf(r8, r8, r2 * r2);
    WriteSum3<float, float>((float*)inout_shared, r59, r75, r31);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r68, r43, r60 * r19);
    r60 = fmaf(r60, r2, r68 * r8);
    r8 = fmaf(r43, r8, r19 * r2);
    WriteSum3<float, float>((float*)inout_shared, r31, r60, r8);
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

void RowFixedRigPinholeResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_calibration,
    unsigned int sensor_calibration_num_alloc,
    SharedIndex* sensor_calibration_indices,
    const float* const sensor_from_rig_log_scale,
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
  RowFixedRigPinholeResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_calibration,
      sensor_calibration_num_alloc,
      sensor_calibration_indices,
      sensor_from_rig_log_scale,
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