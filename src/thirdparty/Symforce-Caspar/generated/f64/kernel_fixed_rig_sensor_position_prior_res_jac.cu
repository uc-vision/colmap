#include "kernel_fixed_rig_sensor_position_prior_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedRigSensorPositionPriorResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        const double* const sensor_from_rig_log_scale,
        double* position,
        unsigned int position_num_alloc,
        double* sqrt_information,
        unsigned int sqrt_information_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
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
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ double out_sensor_from_rig_log_scale_precond_diag_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83;

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
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r5,
                                            r6);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r7,
                                            r8);
    r9 = r5 * r7;
    r10 = 2.00000000000000000e+00;
    r9 = r9 * r10;
    r11 = r6 * r8;
    r12 = -2.00000000000000000e+00;
    r11 = fma(r12, r11, r9);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r6 * r6;
    r14 = r14 * r12;
    r15 = 1.00000000000000000e+00;
    r16 = r5 * r5;
    r16 = fma(r12, r16, r15);
    r17 = r14 + r16;
    r18 = fma(r13, r17, r3 * r11);
    r19 = r6 * r7;
    r19 = r19 * r10;
    r20 = r5 * r8;
    r20 = fma(r10, r20, r19);
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r21);
  };
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = 2.71828182845904523536;
    r22 = pow(r23, r22);
    r21 = r21 * r22;
    r18 = fma(r4, r20, r18);
    r18 = r18 + r21;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r23, r24);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r25, r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = fma(r26, r5, r23 * r8);
    r28 = r24 * r7;
    r27 = fma(r2, r28, r27);
    r27 = fma(r25, r6, r27);
    r28 = r25 * r8;
    r29 = fma(r24, r5, r28);
    r30 = r23 * r6;
    r29 = fma(r26, r7, r29);
    r29 = fma(r2, r30, r29);
    r31 = r10 * r29;
    r32 = r27 * r31;
    r33 = r25 * r5;
    r33 = fma(r2, r33, r24 * r8);
    r33 = fma(r26, r6, r33);
    r33 = fma(r23, r7, r33);
    r34 = fma(r24, r6, r23 * r5);
    r34 = fma(r25, r7, r34);
    r34 = fma(r2, r34, r26 * r8);
    r35 = r12 * r34;
    r36 = fma(r33, r35, r32);
    r37 = r6 * r8;
    r37 = fma(r10, r37, r9);
    r9 = r7 * r8;
    r38 = r5 * r6;
    r38 = r38 * r10;
    r9 = fma(r12, r9, r38);
    r39 = fma(r4, r9, r13 * r37);
    r14 = r15 + r14;
    r40 = r7 * r7;
    r40 = r12 * r40;
    r14 = r14 + r40;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r41,
                                            r42);
    r39 = fma(r3, r14, r39);
    r39 = fma(r41, r22, r39);
    r43 = r29 * r29;
    r43 = r43 * r12;
    r44 = r15 + r43;
    r45 = r12 * r33;
    r45 = r45 * r33;
    r44 = r44 + r45;
    r46 = fma(r44, r39, r36 * r18);
    r47 = r7 * r8;
    r47 = fma(r10, r47, r38);
    r16 = r40 + r16;
    r4 = fma(r4, r16, r3 * r47);
    r3 = r5 * r8;
    r3 = fma(r12, r3, r19);
    r4 = fma(r13, r3, r4);
    r4 = fma(r42, r22, r4);
    r13 = r10 * r27;
    r13 = r13 * r33;
    r19 = fma(r34, r31, r13);
    r46 = fma(r19, r4, r46);
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r40, r38);
    r40 = fma(r40, r2, r2 * r46);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            6 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r46,
                                            r48);
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r49);
    r50 = r10 * r33;
    r50 = fma(r34, r50, r32);
    r32 = r33 * r31;
    r51 = fma(r27, r35, r32);
    r52 = fma(r4, r51, r39 * r50);
    r45 = r15 + r45;
    r53 = r27 * r27;
    r53 = r53 * r12;
    r45 = r45 + r53;
    r52 = fma(r18, r45, r52);
    r52 = fma(r2, r52, r49 * r2);
    r49 = fma(r46, r52, r0 * r40);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            2 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r54,
                                            r55);
    r56 = r10 * r27;
    r56 = fma(r34, r56, r32);
    r13 = fma(r29, r35, r13);
    r32 = fma(r39, r13, r18 * r56);
    r43 = r15 + r43;
    r43 = r43 + r53;
    r32 = fma(r4, r43, r32);
    r38 = fma(r38, r2, r2 * r32);
    r49 = fma(r55, r38, r49);
    r32 = fma(r48, r52, r1 * r40);
    ReadIdx2<1024, double, double, double2>(sqrt_information,
                                            4 * sqrt_information_num_alloc,
                                            global_thread_idx,
                                            r53,
                                            r15);
    r32 = fma(r53, r38, r32);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r49, r32);
    ReadIdx1<1024, double, double, double>(sqrt_information,
                                           8 * sqrt_information_num_alloc,
                                           global_thread_idx,
                                           r57);
    r52 = fma(r57, r52, r54 * r40);
    r52 = fma(r15, r38, r52);
    WriteIdx1<1024, double, double, double>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r52);
    r38 = r2 * r4;
    r40 = r27 * r12;
    r58 = r23 * r8;
    r59 = -5.00000000000000000e-01;
    r60 = r26 * r5;
    r60 = fma(r59, r60, r59 * r58);
    r58 = r25 * r6;
    r60 = fma(r59, r58, r60);
    r61 = r24 * r7;
    r62 = 5.00000000000000000e-01;
    r60 = fma(r62, r61, r60);
    r61 = r23 * r5;
    r58 = r25 * r7;
    r58 = fma(r59, r58, r59 * r61);
    r61 = r26 * r62;
    r63 = r24 * r59;
    r58 = fma(r8, r61, r58);
    r58 = fma(r6, r63, r58);
    r64 = r58 * r35;
    r40 = fma(r60, r40, r64);
    r65 = r10 * r33;
    r66 = r25 * r5;
    r67 = r26 * r6;
    r67 = fma(r59, r67, r62 * r66);
    r66 = r23 * r7;
    r67 = fma(r59, r66, r67);
    r67 = fma(r8, r63, r67);
    r65 = r65 * r67;
    r66 = r24 * r5;
    r66 = fma(r62, r28, r62 * r66);
    r66 = fma(r59, r30, r66);
    r66 = fma(r7, r61, r66);
    r68 = fma(r66, r31, r65);
    r40 = r40 + r68;
    r69 = r2 * r18;
    r70 = r33 * r66;
    r71 = -4.00000000000000000e+00;
    r70 = r70 * r71;
    r72 = r27 * r71;
    r73 = r58 * r72;
    r74 = r70 + r73;
    r69 = fma(r74, r69, r40 * r38);
    r39 = r2 * r39;
    r38 = r10 * r34;
    r74 = r33 * r60;
    r38 = fma(r10, r74, r66 * r38);
    r40 = r10 * r27;
    r75 = r58 * r31;
    r40 = fma(r67, r40, r75);
    r38 = r38 + r40;
    r69 = fma(r38, r39, r69);
    r38 = r2 * r4;
    r76 = r29 * r67;
    r76 = r76 * r71;
    r73 = r76 + r73;
    r77 = r2 * r18;
    r78 = r10 * r27;
    r78 = r78 * r60;
    r79 = r10 * r34;
    r79 = r79 * r58;
    r80 = r78 + r79;
    r68 = r68 + r80;
    r77 = fma(r68, r77, r73 * r38);
    r38 = r10 * r33;
    r38 = r38 * r58;
    r68 = r10 * r27;
    r68 = r68 * r66;
    r73 = r38 + r68;
    r81 = r29 * r12;
    r73 = fma(r60, r81, r73);
    r73 = fma(r67, r35, r73);
    r77 = fma(r73, r39, r77);
    r73 = fma(r55, r77, r46 * r69);
    r81 = r2 * r4;
    r82 = r10 * r34;
    r82 = fma(r67, r82, r68);
    r38 = fma(r60, r31, r38);
    r82 = r82 + r38;
    r68 = r2 * r18;
    r66 = fma(r66, r35, r12 * r74);
    r66 = r66 + r40;
    r68 = fma(r66, r68, r82 * r81);
    r76 = r70 + r76;
    r68 = fma(r76, r39, r68);
    r73 = fma(r0, r68, r73);
    r76 = fma(r53, r77, r48 * r69);
    r76 = fma(r1, r68, r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r73, r76);
    r77 = fma(r15, r77, r57 * r69);
    r77 = fma(r54, r68, r77);
    r68 = r2 * r18;
    r69 = r10 * r33;
    r70 = r23 * r8;
    r81 = r25 * r6;
    r81 = fma(r62, r81, r62 * r70);
    r81 = fma(r5, r61, r81);
    r81 = fma(r7, r63, r81);
    r69 = r69 * r81;
    r70 = r10 * r34;
    r66 = r26 * r7;
    r28 = fma(r59, r28, r59 * r66);
    r28 = fma(r5, r63, r28);
    r28 = fma(r62, r30, r28);
    r70 = fma(r28, r70, r69);
    r70 = r70 + r40;
    r40 = r2 * r4;
    r30 = r29 * r71;
    r30 = r30 * r81;
    r63 = r28 * r72;
    r66 = r30 + r63;
    r40 = fma(r66, r40, r70 * r68);
    r68 = r29 * r12;
    r68 = fma(r81, r35, r67 * r68);
    r66 = r10 * r27;
    r66 = r66 * r58;
    r70 = r10 * r33;
    r70 = fma(r28, r70, r66);
    r68 = r68 + r70;
    r40 = fma(r68, r39, r40);
    r68 = r2 * r4;
    r82 = r27 * r12;
    r82 = fma(r67, r82, r69);
    r82 = r82 + r75;
    r82 = fma(r28, r35, r82);
    r75 = r2 * r18;
    r69 = r33 * r58;
    r69 = r69 * r71;
    r63 = r69 + r63;
    r75 = fma(r63, r75, r82 * r68);
    r79 = r65 + r79;
    r65 = r10 * r27;
    r68 = r28 * r31;
    r65 = fma(r81, r65, r68);
    r79 = r79 + r65;
    r75 = fma(r79, r39, r75);
    r79 = fma(r46, r75, r55 * r40);
    r63 = r2 * r4;
    r82 = r10 * r34;
    r82 = fma(r67, r31, r81 * r82);
    r82 = r82 + r70;
    r81 = r2 * r18;
    r83 = r12 * r33;
    r83 = fma(r67, r83, r64);
    r83 = r83 + r65;
    r81 = fma(r83, r81, r82 * r63);
    r69 = r30 + r69;
    r81 = fma(r69, r39, r81);
    r79 = fma(r0, r81, r79);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r77, r79);
    r69 = fma(r48, r75, r53 * r40);
    r69 = fma(r1, r81, r69);
    r75 = fma(r57, r75, r15 * r40);
    r75 = fma(r54, r81, r75);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r69, r75);
    r81 = r2 * r18;
    r40 = r12 * r33;
    r40 = fma(r28, r40, r66);
    r66 = r24 * r8;
    r30 = r25 * r5;
    r30 = fma(r59, r30, r62 * r66);
    r66 = r23 * r7;
    r30 = fma(r62, r66, r30);
    r30 = fma(r6, r61, r30);
    r31 = r30 * r31;
    r40 = r40 + r31;
    r40 = fma(r60, r35, r40);
    r61 = r2 * r4;
    r66 = r10 * r33;
    r66 = r66 * r30;
    r68 = r66 + r68;
    r68 = r68 + r80;
    r61 = fma(r68, r61, r40 * r81);
    r58 = r29 * r58;
    r58 = r58 * r71;
    r74 = r71 * r74;
    r71 = r58 + r74;
    r61 = fma(r71, r39, r61);
    r71 = r2 * r18;
    r72 = r30 * r72;
    r74 = r74 + r72;
    r81 = r2 * r4;
    r68 = r27 * r12;
    r35 = fma(r30, r35, r28 * r68);
    r35 = r35 + r38;
    r81 = fma(r35, r81, r74 * r71);
    r71 = r10 * r34;
    r71 = fma(r60, r71, r31);
    r71 = r71 + r70;
    r81 = fma(r71, r39, r81);
    r71 = fma(r46, r81, r0 * r61);
    r70 = r2 * r18;
    r31 = r10 * r27;
    r60 = r10 * r34;
    r60 = fma(r30, r60, r28 * r31);
    r60 = r60 + r38;
    r38 = r2 * r4;
    r72 = r58 + r72;
    r38 = fma(r72, r38, r60 * r70);
    r66 = r78 + r66;
    r78 = r29 * r12;
    r66 = fma(r28, r78, r66);
    r66 = r66 + r64;
    r38 = fma(r66, r39, r38);
    r71 = fma(r55, r38, r71);
    r39 = fma(r48, r81, r1 * r61);
    r39 = fma(r53, r38, r39);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r71, r39);
    r81 = fma(r57, r81, r54 * r61);
    r81 = fma(r15, r38, r81);
    r38 = r2 * r11;
    r61 = r2 * r47;
    r61 = fma(r51, r61, r45 * r38);
    r38 = r2 * r14;
    r61 = fma(r50, r38, r61);
    r38 = r2 * r47;
    r66 = r2 * r11;
    r66 = fma(r56, r66, r43 * r38);
    r38 = r2 * r14;
    r66 = fma(r13, r38, r66);
    r38 = fma(r55, r66, r46 * r61);
    r64 = r2 * r44;
    r78 = r2 * r36;
    r78 = fma(r11, r78, r14 * r64);
    r64 = r2 * r19;
    r78 = fma(r47, r64, r78);
    r38 = fma(r0, r78, r38);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r81, r38);
    r64 = fma(r53, r66, r48 * r61);
    r64 = fma(r1, r78, r64);
    r66 = fma(r15, r66, r57 * r61);
    r66 = fma(r54, r78, r66);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r64, r66);
    r78 = r2 * r16;
    r61 = r2 * r20;
    r61 = fma(r56, r61, r43 * r78);
    r78 = r2 * r9;
    r61 = fma(r13, r78, r61);
    r78 = r2 * r20;
    r28 = r2 * r16;
    r28 = fma(r51, r28, r45 * r78);
    r78 = r2 * r9;
    r28 = fma(r50, r78, r28);
    r78 = fma(r46, r28, r55 * r61);
    r70 = r2 * r44;
    r72 = r2 * r19;
    r72 = fma(r16, r72, r9 * r70);
    r70 = r2 * r36;
    r72 = fma(r20, r70, r72);
    r78 = fma(r0, r72, r78);
    r70 = fma(r48, r28, r53 * r61);
    r70 = fma(r1, r72, r70);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 12 * out_pose_jac_num_alloc, global_thread_idx, r78, r70);
    r28 = fma(r57, r28, r15 * r61);
    r28 = fma(r54, r72, r28);
    r72 = r2 * r17;
    r61 = r2 * r3;
    r61 = fma(r51, r61, r45 * r72);
    r72 = r2 * r37;
    r61 = fma(r50, r72, r61);
    r72 = r2 * r44;
    r60 = r2 * r36;
    r60 = fma(r17, r60, r37 * r72);
    r72 = r2 * r19;
    r60 = fma(r3, r72, r60);
    r72 = fma(r0, r60, r46 * r61);
    r58 = r2 * r3;
    r31 = r2 * r17;
    r31 = fma(r56, r31, r43 * r58);
    r58 = r2 * r37;
    r31 = fma(r13, r58, r31);
    r72 = fma(r55, r31, r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 14 * out_pose_jac_num_alloc, global_thread_idx, r28, r72);
    r58 = fma(r1, r60, r48 * r61);
    r58 = fma(r53, r31, r58);
    r60 = fma(r54, r60, r57 * r61);
    r60 = fma(r15, r31, r60);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 16 * out_pose_jac_num_alloc, global_thread_idx, r58, r60);
    r31 = r2 * r52;
    r61 = r2 * r49;
    r61 = fma(r73, r61, r77 * r31);
    r31 = r2 * r32;
    r61 = fma(r76, r31, r61);
    r31 = r2 * r52;
    r30 = r2 * r49;
    r30 = fma(r79, r30, r75 * r31);
    r31 = r2 * r32;
    r30 = fma(r69, r31, r30);
    WriteSum2<double, double>((double*)inout_shared, r61, r30);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r2 * r52;
    r61 = r2 * r49;
    r61 = fma(r71, r61, r81 * r30);
    r30 = r2 * r32;
    r61 = fma(r39, r30, r61);
    r30 = r2 * r32;
    r31 = r2 * r49;
    r31 = fma(r38, r31, r64 * r30);
    r30 = r2 * r52;
    r31 = fma(r66, r30, r31);
    WriteSum2<double, double>((double*)inout_shared, r61, r31);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r2 * r32;
    r61 = r2 * r49;
    r61 = fma(r78, r61, r70 * r31);
    r31 = r2 * r52;
    r61 = fma(r28, r31, r61);
    r31 = r2 * r32;
    r30 = r2 * r52;
    r30 = fma(r60, r30, r58 * r31);
    r31 = r2 * r49;
    r30 = fma(r72, r31, r30);
    WriteSum2<double, double>((double*)inout_shared, r61, r30);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r77, r77, r76 * r76);
    r30 = fma(r73, r73, r30);
    r61 = fma(r79, r79, r69 * r69);
    r61 = fma(r75, r75, r61);
    WriteSum2<double, double>((double*)inout_shared, r30, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r81, r81, r39 * r39);
    r61 = fma(r71, r71, r61);
    r30 = fma(r66, r66, r64 * r64);
    r30 = fma(r38, r38, r30);
    WriteSum2<double, double>((double*)inout_shared, r61, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r78, r78, r70 * r70);
    r30 = fma(r28, r28, r30);
    r61 = fma(r60, r60, r58 * r58);
    r61 = fma(r72, r72, r61);
    WriteSum2<double, double>((double*)inout_shared, r30, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r73, r79, r76 * r69);
    r61 = fma(r77, r75, r61);
    r30 = fma(r73, r71, r76 * r39);
    r30 = fma(r77, r81, r30);
    WriteSum2<double, double>((double*)inout_shared, r61, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r76, r64, r77 * r66);
    r30 = fma(r73, r38, r30);
    r61 = fma(r73, r78, r76 * r70);
    r61 = fma(r77, r28, r61);
    WriteSum2<double, double>((double*)inout_shared, r30, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r73, r72, r77 * r60);
    r73 = fma(r76, r58, r73);
    r76 = fma(r75, r81, r79 * r71);
    r76 = fma(r69, r39, r76);
    WriteSum2<double, double>((double*)inout_shared, r73, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r75, r66, r69 * r64);
    r76 = fma(r79, r38, r76);
    r73 = fma(r69, r70, r79 * r78);
    r73 = fma(r75, r28, r73);
    WriteSum2<double, double>((double*)inout_shared, r76, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r75, r60, r79 * r72);
    r75 = fma(r69, r58, r75);
    r69 = fma(r71, r38, r81 * r66);
    r69 = fma(r39, r64, r69);
    WriteSum2<double, double>((double*)inout_shared, r75, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r39, r70, r81 * r28);
    r69 = fma(r71, r78, r69);
    r39 = fma(r39, r58, r71 * r72);
    r39 = fma(r81, r60, r39);
    WriteSum2<double, double>((double*)inout_shared, r69, r39);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r64, r70, r38 * r78);
    r39 = fma(r66, r28, r39);
    r38 = fma(r38, r72, r66 * r60);
    r38 = fma(r64, r58, r38);
    WriteSum2<double, double>((double*)inout_shared, r39, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r78, r72, r28 * r60);
    r72 = fma(r70, r58, r72);
    WriteSum1<double, double>((double*)inout_shared, r72);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r41 * r2;
    r72 = r72 * r22;
    r58 = r42 * r2;
    r58 = r58 * r22;
    r58 = fma(r51, r58, r50 * r72);
    r21 = r2 * r21;
    r58 = fma(r45, r21, r58);
    r45 = r42 * r2;
    r45 = r45 * r22;
    r72 = r41 * r2;
    r72 = r72 * r22;
    r72 = fma(r13, r72, r43 * r45);
    r72 = fma(r56, r21, r72);
    r55 = fma(r55, r72, r46 * r58);
    r46 = r42 * r2;
    r46 = r46 * r22;
    r56 = r41 * r2;
    r56 = r56 * r22;
    r56 = fma(r44, r56, r19 * r46);
    r56 = fma(r36, r21, r56);
    r55 = fma(r0, r56, r55);
    r53 = fma(r53, r72, r48 * r58);
    r53 = fma(r1, r56, r53);
    WriteIdx2<1024, double, double, double2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r55,
        r53);
    r72 = fma(r15, r72, r57 * r58);
    r72 = fma(r54, r56, r72);
    WriteIdx1<1024, double, double, double>(
        out_sensor_from_rig_log_scale_jac,
        2 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r72);
    r56 = r2 * r32;
    r54 = r2 * r49;
    r54 = fma(r55, r54, r53 * r56);
    r56 = r2 * r52;
    r54 = fma(r72, r56, r54);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r54);
  if (global_thread_idx < problem_size) {
    r72 = fma(r72, r72, r55 * r55);
    r72 = fma(r53, r53, r72);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r72);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_njtr_local,
                        out_sensor_from_rig_log_scale_njtr,
                        1);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                        out_sensor_from_rig_log_scale_precond_diag,
                        1);
}

void FixedRigSensorPositionPriorResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const double* const sensor_from_rig_log_scale,
    double* position,
    unsigned int position_num_alloc,
    double* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigSensorPositionPriorResJacKernel<<<n_blocks, 1024>>>(
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