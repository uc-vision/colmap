#include "kernel_fixed_rig_sensor_position_prior_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigSensorPositionPriorResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        const float* const sensor_from_rig_log_scale,
        float* position,
        unsigned int position_num_alloc,
        float* sqrt_information,
        unsigned int sqrt_information_num_alloc,
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
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_precond_diag_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83;

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
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10,
                                         r11);
    r12 = r8 * r10;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = r9 * r11;
    r15 = -2.00000000000000000e+00;
    r14 = fmaf(r15, r14, r12);
    r16 = r9 * r9;
    r16 = r16 * r15;
    r17 = 1.00000000000000000e+00;
    r18 = r8 * r8;
    r18 = fmaf(r15, r18, r17);
    r19 = r16 + r18;
    r20 = fmaf(r7, r19, r5 * r14);
    r21 = r9 * r10;
    r21 = r21 * r13;
    r22 = r8 * r11;
    r22 = fmaf(r13, r22, r21);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r23,
                                         r24,
                                         r25);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = 2.71828182845904523536;
    r26 = powf(r27, r26);
    r25 = r25 * r26;
    r20 = fmaf(r6, r22, r20);
    r20 = r20 + r25;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r27,
                       r28,
                       r29,
                       r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r30, r8, r27 * r11);
    r32 = r28 * r10;
    r31 = fmaf(r4, r32, r31);
    r31 = fmaf(r29, r9, r31);
    r32 = r29 * r11;
    r33 = fmaf(r28, r8, r32);
    r34 = r27 * r9;
    r33 = fmaf(r30, r10, r33);
    r33 = fmaf(r4, r34, r33);
    r35 = r13 * r33;
    r36 = r31 * r35;
    r37 = r29 * r8;
    r37 = fmaf(r4, r37, r28 * r11);
    r37 = fmaf(r30, r9, r37);
    r37 = fmaf(r27, r10, r37);
    r38 = fmaf(r28, r9, r27 * r8);
    r38 = fmaf(r29, r10, r38);
    r38 = fmaf(r4, r38, r30 * r11);
    r39 = r15 * r38;
    r40 = fmaf(r37, r39, r36);
    r41 = r9 * r11;
    r41 = fmaf(r13, r41, r12);
    r12 = r10 * r11;
    r42 = r8 * r9;
    r42 = r42 * r13;
    r12 = fmaf(r15, r12, r42);
    r43 = fmaf(r6, r12, r7 * r41);
    r16 = r17 + r16;
    r44 = r10 * r10;
    r44 = r15 * r44;
    r16 = r16 + r44;
    r43 = fmaf(r5, r16, r43);
    r43 = fmaf(r23, r26, r43);
    r45 = r33 * r33;
    r45 = r45 * r15;
    r46 = r17 + r45;
    r47 = r15 * r37;
    r47 = r47 * r37;
    r46 = r46 + r47;
    r48 = fmaf(r46, r43, r40 * r20);
    r49 = r10 * r11;
    r49 = fmaf(r13, r49, r42);
    r18 = r44 + r18;
    r6 = fmaf(r6, r18, r5 * r49);
    r5 = r8 * r11;
    r5 = fmaf(r15, r5, r21);
    r6 = fmaf(r7, r5, r6);
    r6 = fmaf(r24, r26, r6);
    r7 = r13 * r31;
    r7 = r7 * r37;
    r21 = fmaf(r38, r35, r7);
    r48 = fmaf(r21, r6, r48);
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r44, r42, r50);
    r44 = fmaf(r44, r4, r4 * r48);
    ReadIdx4<1024, float, float, float4>(sqrt_information,
                                         4 * sqrt_information_num_alloc,
                                         global_thread_idx,
                                         r48,
                                         r51,
                                         r52,
                                         r53);
    r54 = r13 * r37;
    r54 = fmaf(r38, r54, r36);
    r36 = r37 * r35;
    r55 = fmaf(r31, r39, r36);
    r56 = fmaf(r6, r55, r43 * r54);
    r47 = r17 + r47;
    r57 = r31 * r31;
    r57 = r57 * r15;
    r47 = r47 + r57;
    r56 = fmaf(r20, r47, r56);
    r56 = fmaf(r4, r56, r50 * r4);
    r50 = fmaf(r52, r56, r0 * r44);
    r58 = r13 * r31;
    r58 = fmaf(r38, r58, r36);
    r7 = fmaf(r33, r39, r7);
    r36 = fmaf(r43, r7, r20 * r58);
    r45 = r17 + r45;
    r45 = r45 + r57;
    r36 = fmaf(r6, r45, r36);
    r42 = fmaf(r42, r4, r4 * r36);
    r50 = fmaf(r3, r42, r50);
    r36 = fmaf(r53, r56, r1 * r44);
    r36 = fmaf(r48, r42, r36);
    ReadIdx1<1024, float, float, float>(sqrt_information,
                                        8 * sqrt_information_num_alloc,
                                        global_thread_idx,
                                        r57);
    r56 = fmaf(r57, r56, r2 * r44);
    r56 = fmaf(r51, r42, r56);
    WriteIdx3<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r50, r36, r56);
    r42 = fmaf(r50, r50, r56 * r56);
    r42 = fmaf(r36, r36, r42);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r42);
  if (global_thread_idx < problem_size) {
    r42 = r4 * r6;
    r44 = r31 * r15;
    r17 = r27 * r11;
    r59 = -5.00000000000000000e-01;
    r60 = r30 * r8;
    r60 = fmaf(r59, r60, r59 * r17);
    r17 = r29 * r9;
    r60 = fmaf(r59, r17, r60);
    r61 = r28 * r10;
    r62 = 5.00000000000000000e-01;
    r60 = fmaf(r62, r61, r60);
    r61 = r27 * r8;
    r17 = r29 * r10;
    r17 = fmaf(r59, r17, r59 * r61);
    r61 = r30 * r62;
    r63 = r28 * r59;
    r17 = fmaf(r11, r61, r17);
    r17 = fmaf(r9, r63, r17);
    r64 = r17 * r39;
    r44 = fmaf(r60, r44, r64);
    r65 = r13 * r37;
    r66 = r29 * r8;
    r67 = r30 * r9;
    r67 = fmaf(r59, r67, r62 * r66);
    r66 = r27 * r10;
    r67 = fmaf(r59, r66, r67);
    r67 = fmaf(r11, r63, r67);
    r65 = r65 * r67;
    r66 = r28 * r8;
    r66 = fmaf(r62, r32, r62 * r66);
    r66 = fmaf(r59, r34, r66);
    r66 = fmaf(r10, r61, r66);
    r68 = fmaf(r66, r35, r65);
    r44 = r44 + r68;
    r69 = r4 * r20;
    r70 = r37 * r66;
    r71 = -4.00000000000000000e+00;
    r70 = r70 * r71;
    r72 = r31 * r71;
    r73 = r17 * r72;
    r74 = r70 + r73;
    r69 = fmaf(r74, r69, r44 * r42);
    r43 = r4 * r43;
    r42 = r13 * r38;
    r74 = r37 * r60;
    r42 = fmaf(r13, r74, r66 * r42);
    r44 = r13 * r31;
    r75 = r17 * r35;
    r44 = fmaf(r67, r44, r75);
    r42 = r42 + r44;
    r69 = fmaf(r42, r43, r69);
    r42 = r4 * r6;
    r76 = r33 * r67;
    r76 = r76 * r71;
    r73 = r76 + r73;
    r77 = r4 * r20;
    r78 = r13 * r31;
    r78 = r78 * r60;
    r79 = r13 * r38;
    r79 = r79 * r17;
    r80 = r78 + r79;
    r68 = r68 + r80;
    r77 = fmaf(r68, r77, r73 * r42);
    r42 = r13 * r37;
    r42 = r42 * r17;
    r68 = r13 * r31;
    r68 = r68 * r66;
    r73 = r42 + r68;
    r81 = r33 * r15;
    r73 = fmaf(r60, r81, r73);
    r73 = fmaf(r67, r39, r73);
    r77 = fmaf(r73, r43, r77);
    r73 = fmaf(r3, r77, r52 * r69);
    r81 = r4 * r6;
    r82 = r13 * r38;
    r82 = fmaf(r67, r82, r68);
    r42 = fmaf(r60, r35, r42);
    r82 = r82 + r42;
    r68 = r4 * r20;
    r66 = fmaf(r66, r39, r15 * r74);
    r66 = r66 + r44;
    r68 = fmaf(r66, r68, r82 * r81);
    r76 = r70 + r76;
    r68 = fmaf(r76, r43, r68);
    r73 = fmaf(r0, r68, r73);
    r76 = fmaf(r48, r77, r53 * r69);
    r76 = fmaf(r1, r68, r76);
    r77 = fmaf(r51, r77, r57 * r69);
    r77 = fmaf(r2, r68, r77);
    r68 = r4 * r20;
    r69 = r13 * r37;
    r70 = r27 * r11;
    r81 = r29 * r9;
    r81 = fmaf(r62, r81, r62 * r70);
    r81 = fmaf(r8, r61, r81);
    r81 = fmaf(r10, r63, r81);
    r69 = r69 * r81;
    r70 = r13 * r38;
    r66 = r30 * r10;
    r32 = fmaf(r59, r32, r59 * r66);
    r32 = fmaf(r8, r63, r32);
    r32 = fmaf(r62, r34, r32);
    r70 = fmaf(r32, r70, r69);
    r70 = r70 + r44;
    r44 = r4 * r6;
    r34 = r33 * r71;
    r34 = r34 * r81;
    r63 = r32 * r72;
    r66 = r34 + r63;
    r44 = fmaf(r66, r44, r70 * r68);
    r68 = r33 * r15;
    r68 = fmaf(r81, r39, r67 * r68);
    r66 = r13 * r31;
    r66 = r66 * r17;
    r70 = r13 * r37;
    r70 = fmaf(r32, r70, r66);
    r68 = r68 + r70;
    r44 = fmaf(r68, r43, r44);
    r68 = r4 * r6;
    r82 = r31 * r15;
    r82 = fmaf(r67, r82, r69);
    r82 = r82 + r75;
    r82 = fmaf(r32, r39, r82);
    r75 = r4 * r20;
    r69 = r37 * r17;
    r69 = r69 * r71;
    r63 = r69 + r63;
    r75 = fmaf(r63, r75, r82 * r68);
    r79 = r65 + r79;
    r65 = r13 * r31;
    r68 = r32 * r35;
    r65 = fmaf(r81, r65, r68);
    r79 = r79 + r65;
    r75 = fmaf(r79, r43, r75);
    r79 = fmaf(r52, r75, r3 * r44);
    r63 = r4 * r6;
    r82 = r13 * r38;
    r82 = fmaf(r67, r35, r81 * r82);
    r82 = r82 + r70;
    r81 = r4 * r20;
    r83 = r15 * r37;
    r83 = fmaf(r67, r83, r64);
    r83 = r83 + r65;
    r81 = fmaf(r83, r81, r82 * r63);
    r69 = r34 + r69;
    r81 = fmaf(r69, r43, r81);
    r79 = fmaf(r0, r81, r79);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r73,
                                          r76,
                                          r77,
                                          r79);
    r69 = fmaf(r53, r75, r48 * r44);
    r69 = fmaf(r1, r81, r69);
    r75 = fmaf(r57, r75, r51 * r44);
    r75 = fmaf(r2, r81, r75);
    r81 = r4 * r20;
    r44 = r15 * r37;
    r44 = fmaf(r32, r44, r66);
    r66 = r28 * r11;
    r34 = r29 * r8;
    r34 = fmaf(r59, r34, r62 * r66);
    r66 = r27 * r10;
    r34 = fmaf(r62, r66, r34);
    r34 = fmaf(r9, r61, r34);
    r35 = r34 * r35;
    r44 = r44 + r35;
    r44 = fmaf(r60, r39, r44);
    r61 = r4 * r6;
    r66 = r13 * r37;
    r66 = r66 * r34;
    r68 = r66 + r68;
    r68 = r68 + r80;
    r61 = fmaf(r68, r61, r44 * r81);
    r17 = r33 * r17;
    r17 = r17 * r71;
    r74 = r71 * r74;
    r71 = r17 + r74;
    r61 = fmaf(r71, r43, r61);
    r71 = r4 * r20;
    r72 = r34 * r72;
    r74 = r74 + r72;
    r81 = r4 * r6;
    r68 = r31 * r15;
    r39 = fmaf(r34, r39, r32 * r68);
    r39 = r39 + r42;
    r81 = fmaf(r39, r81, r74 * r71);
    r71 = r13 * r38;
    r71 = fmaf(r60, r71, r35);
    r71 = r71 + r70;
    r81 = fmaf(r71, r43, r81);
    r71 = fmaf(r52, r81, r0 * r61);
    r70 = r4 * r20;
    r35 = r13 * r31;
    r60 = r13 * r38;
    r60 = fmaf(r34, r60, r32 * r35);
    r60 = r60 + r42;
    r42 = r4 * r6;
    r72 = r17 + r72;
    r42 = fmaf(r72, r42, r60 * r70);
    r66 = r78 + r66;
    r78 = r33 * r15;
    r66 = fmaf(r32, r78, r66);
    r66 = r66 + r64;
    r42 = fmaf(r66, r43, r42);
    r71 = fmaf(r3, r42, r71);
    r43 = fmaf(r53, r81, r1 * r61);
    r43 = fmaf(r48, r42, r43);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r69,
                                          r75,
                                          r71,
                                          r43);
    r81 = fmaf(r57, r81, r2 * r61);
    r81 = fmaf(r51, r42, r81);
    r42 = r4 * r14;
    r61 = r4 * r49;
    r61 = fmaf(r55, r61, r47 * r42);
    r42 = r4 * r16;
    r61 = fmaf(r54, r42, r61);
    r42 = r4 * r49;
    r66 = r4 * r14;
    r66 = fmaf(r58, r66, r45 * r42);
    r42 = r4 * r16;
    r66 = fmaf(r7, r42, r66);
    r42 = fmaf(r3, r66, r52 * r61);
    r64 = r4 * r46;
    r78 = r4 * r40;
    r78 = fmaf(r14, r78, r16 * r64);
    r64 = r4 * r21;
    r78 = fmaf(r49, r64, r78);
    r42 = fmaf(r0, r78, r42);
    r64 = fmaf(r48, r66, r53 * r61);
    r64 = fmaf(r1, r78, r64);
    r66 = fmaf(r51, r66, r57 * r61);
    r66 = fmaf(r2, r78, r66);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r81,
                                          r42,
                                          r64,
                                          r66);
    r78 = r4 * r18;
    r61 = r4 * r22;
    r61 = fmaf(r58, r61, r45 * r78);
    r78 = r4 * r12;
    r61 = fmaf(r7, r78, r61);
    r78 = r4 * r22;
    r32 = r4 * r18;
    r32 = fmaf(r55, r32, r47 * r78);
    r78 = r4 * r12;
    r32 = fmaf(r54, r78, r32);
    r78 = fmaf(r52, r32, r3 * r61);
    r70 = r4 * r46;
    r72 = r4 * r21;
    r72 = fmaf(r18, r72, r12 * r70);
    r70 = r4 * r40;
    r72 = fmaf(r22, r70, r72);
    r78 = fmaf(r0, r72, r78);
    r70 = fmaf(r53, r32, r48 * r61);
    r70 = fmaf(r1, r72, r70);
    r32 = fmaf(r57, r32, r51 * r61);
    r32 = fmaf(r2, r72, r32);
    r72 = r4 * r19;
    r61 = r4 * r5;
    r61 = fmaf(r55, r61, r47 * r72);
    r72 = r4 * r41;
    r61 = fmaf(r54, r72, r61);
    r72 = r4 * r46;
    r60 = r4 * r40;
    r60 = fmaf(r19, r60, r41 * r72);
    r72 = r4 * r21;
    r60 = fmaf(r5, r72, r60);
    r72 = fmaf(r0, r60, r52 * r61);
    r17 = r4 * r5;
    r35 = r4 * r19;
    r35 = fmaf(r58, r35, r45 * r17);
    r17 = r4 * r41;
    r35 = fmaf(r7, r17, r35);
    r72 = fmaf(r3, r35, r72);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          12 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r78,
                                          r70,
                                          r32,
                                          r72);
    r17 = fmaf(r1, r60, r53 * r61);
    r17 = fmaf(r48, r35, r17);
    r60 = fmaf(r2, r60, r57 * r61);
    r60 = fmaf(r51, r35, r60);
    WriteIdx2<1024, float, float, float2>(
        out_pose_jac, 16 * out_pose_jac_num_alloc, global_thread_idx, r17, r60);
    r35 = r4 * r56;
    r61 = r4 * r50;
    r61 = fmaf(r73, r61, r77 * r35);
    r35 = r4 * r36;
    r61 = fmaf(r76, r35, r61);
    r35 = r4 * r56;
    r34 = r4 * r50;
    r34 = fmaf(r79, r34, r75 * r35);
    r35 = r4 * r36;
    r34 = fmaf(r69, r35, r34);
    r35 = r4 * r56;
    r39 = r4 * r50;
    r39 = fmaf(r71, r39, r81 * r35);
    r35 = r4 * r36;
    r39 = fmaf(r43, r35, r39);
    r35 = r4 * r36;
    r74 = r4 * r50;
    r74 = fmaf(r42, r74, r64 * r35);
    r35 = r4 * r56;
    r74 = fmaf(r66, r35, r74);
    WriteSum4<float, float>((float*)inout_shared, r61, r34, r39, r74);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r4 * r36;
    r39 = r4 * r50;
    r39 = fmaf(r78, r39, r70 * r74);
    r74 = r4 * r56;
    r39 = fmaf(r32, r74, r39);
    r74 = r4 * r36;
    r34 = r4 * r56;
    r34 = fmaf(r60, r34, r17 * r74);
    r74 = r4 * r50;
    r34 = fmaf(r72, r74, r34);
    WriteSum2<float, float>((float*)inout_shared, r39, r34);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r77, r77, r76 * r76);
    r34 = fmaf(r73, r73, r34);
    r39 = fmaf(r79, r79, r69 * r69);
    r39 = fmaf(r75, r75, r39);
    r74 = fmaf(r81, r81, r43 * r43);
    r74 = fmaf(r71, r71, r74);
    r61 = fmaf(r66, r66, r64 * r64);
    r61 = fmaf(r42, r42, r61);
    WriteSum4<float, float>((float*)inout_shared, r34, r39, r74, r61);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r78, r78, r70 * r70);
    r61 = fmaf(r32, r32, r61);
    r74 = fmaf(r60, r60, r17 * r17);
    r74 = fmaf(r72, r72, r74);
    WriteSum2<float, float>((float*)inout_shared, r61, r74);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r73, r79, r76 * r69);
    r74 = fmaf(r77, r75, r74);
    r61 = fmaf(r73, r71, r76 * r43);
    r61 = fmaf(r77, r81, r61);
    r39 = fmaf(r76, r64, r77 * r66);
    r39 = fmaf(r73, r42, r39);
    r34 = fmaf(r73, r78, r76 * r70);
    r34 = fmaf(r77, r32, r34);
    WriteSum4<float, float>((float*)inout_shared, r74, r61, r39, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r73, r72, r77 * r60);
    r73 = fmaf(r76, r17, r73);
    r76 = fmaf(r75, r81, r79 * r71);
    r76 = fmaf(r69, r43, r76);
    r77 = fmaf(r75, r66, r69 * r64);
    r77 = fmaf(r79, r42, r77);
    r34 = fmaf(r69, r70, r79 * r78);
    r34 = fmaf(r75, r32, r34);
    WriteSum4<float, float>((float*)inout_shared, r73, r76, r77, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r75, r60, r79 * r72);
    r75 = fmaf(r69, r17, r75);
    r69 = fmaf(r71, r42, r81 * r66);
    r69 = fmaf(r43, r64, r69);
    r79 = fmaf(r43, r70, r81 * r32);
    r79 = fmaf(r71, r78, r79);
    r43 = fmaf(r43, r17, r71 * r72);
    r43 = fmaf(r81, r60, r43);
    WriteSum4<float, float>((float*)inout_shared, r75, r69, r79, r43);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fmaf(r64, r70, r42 * r78);
    r43 = fmaf(r66, r32, r43);
    r42 = fmaf(r42, r72, r66 * r60);
    r42 = fmaf(r64, r17, r42);
    r72 = fmaf(r78, r72, r32 * r60);
    r72 = fmaf(r70, r17, r72);
    WriteSum3<float, float>((float*)inout_shared, r43, r42, r72);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r23 * r4;
    r72 = r72 * r26;
    r42 = r24 * r4;
    r42 = r42 * r26;
    r42 = fmaf(r55, r42, r54 * r72);
    r25 = r4 * r25;
    r42 = fmaf(r47, r25, r42);
    r47 = r24 * r4;
    r47 = r47 * r26;
    r72 = r23 * r4;
    r72 = r72 * r26;
    r72 = fmaf(r7, r72, r45 * r47);
    r72 = fmaf(r58, r25, r72);
    r3 = fmaf(r3, r72, r52 * r42);
    r52 = r24 * r4;
    r52 = r52 * r26;
    r58 = r23 * r4;
    r58 = r58 * r26;
    r58 = fmaf(r46, r58, r21 * r52);
    r58 = fmaf(r40, r25, r58);
    r3 = fmaf(r0, r58, r3);
    r48 = fmaf(r48, r72, r53 * r42);
    r48 = fmaf(r1, r58, r48);
    r72 = fmaf(r51, r72, r57 * r42);
    r72 = fmaf(r2, r58, r72);
    WriteIdx3<1024, float, float, float4>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r3,
        r48,
        r72);
    r58 = r4 * r36;
    r2 = r4 * r50;
    r2 = fmaf(r3, r2, r48 * r58);
    r58 = r4 * r56;
    r2 = fmaf(r72, r58, r2);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  if (global_thread_idx < problem_size) {
    r72 = fmaf(r72, r72, r3 * r3);
    r72 = fmaf(r48, r48, r72);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r72);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                       out_sensor_from_rig_log_scale_precond_diag,
                       1);
}

void FixedRigSensorPositionPriorResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* position,
    unsigned int position_num_alloc,
    float* sqrt_information,
    unsigned int sqrt_information_num_alloc,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigSensorPositionPriorResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      sensor_from_rig_log_scale,
      position,
      position_num_alloc,
      sqrt_information,
      sqrt_information_num_alloc,
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
      problem_size);
}

}  // namespace caspar