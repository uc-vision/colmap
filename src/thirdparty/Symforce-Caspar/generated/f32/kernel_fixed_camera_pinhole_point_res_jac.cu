#include "kernel_fixed_camera_pinhole_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedCameraPinholePointResJacKernel(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex image_from_world_indices_loc[1024];
  image_from_world_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? image_from_world_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r1);
    r2 = -1.00000000000000000e+00;
    r3 = 9.99999999999999955e-07;
  };
  LoadShared<4, float, float>(image_from_world,
                              8 * image_from_world_num_alloc,
                              image_from_world_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       image_from_world_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10);
  };
  __syncthreads();
  LoadShared<4, float, float>(image_from_world,
                              0 * image_from_world_num_alloc,
                              image_from_world_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       image_from_world_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r8, r13, r7);
  };
  LoadShared<4, float, float>(image_from_world,
                              4 * image_from_world_num_alloc,
                              image_from_world_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       image_from_world_indices_loc[threadIdx.x].target,
                       r15,
                       r16,
                       r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r10, r4, r7);
    r7 = fmaf(r9, r16, r7);
    r19 = copysign(1.0, r7);
    r19 = fmaf(r3, r19, r7);
    r3 = 1.0 / r19;
    r5 = fmaf(r8, r11, r5);
    r5 = fmaf(r9, r14, r5);
    r5 = fmaf(r10, r17, r5);
    r0 = fmaf(r5, r3, r0 * r2);
    r8 = fmaf(r8, r12, r6);
    r8 = fmaf(r10, r18, r8);
    r8 = fmaf(r9, r15, r8);
    r1 = fmaf(r8, r3, r1 * r2);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r9 = r2 * r1;
    r19 = r19 * r19;
    r19 = 1.0 / r19;
    r19 = r2 * r19;
    r8 = r8 * r19;
    r12 = fmaf(r13, r8, r12 * r3);
    r10 = r2 * r0;
    r6 = r13 * r5;
    r6 = fmaf(r19, r6, r11 * r3);
    r10 = fmaf(r6, r10, r12 * r9);
    r9 = r2 * r1;
    r15 = fmaf(r16, r8, r15 * r3);
    r11 = r2 * r0;
    r7 = r16 * r5;
    r7 = fmaf(r19, r7, r14 * r3);
    r11 = fmaf(r7, r11, r15 * r9);
    r9 = r2 * r1;
    r8 = fmaf(r4, r8, r18 * r3);
    r18 = r2 * r0;
    r14 = r4 * r5;
    r14 = fmaf(r19, r14, r17 * r3);
    r18 = fmaf(r14, r18, r8 * r9);
    WriteSum3<float, float>((float*)inout_shared, r10, r11, r18);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fmaf(r6, r6, r12 * r12);
    r11 = fmaf(r7, r7, r15 * r15);
    r10 = fmaf(r14, r14, r8 * r8);
    WriteSum3<float, float>((float*)inout_shared, r18, r11, r10);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r6, r7, r12 * r15);
    r6 = fmaf(r6, r14, r12 * r8);
    r14 = fmaf(r7, r14, r15 * r8);
    WriteSum3<float, float>((float*)inout_shared, r10, r6, r14);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void FixedCameraPinholePointResJac(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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
  FixedCameraPinholePointResJacKernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      image_from_world,
      image_from_world_num_alloc,
      image_from_world_indices,
      pixel,
      pixel_num_alloc,
      out_res,
      out_res_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar