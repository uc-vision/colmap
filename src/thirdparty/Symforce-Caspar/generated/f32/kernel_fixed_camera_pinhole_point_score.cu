#include "kernel_fixed_camera_pinhole_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedCameraPinholePointScoreKernel(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    const float* const reprojection_loss_scale,
    float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17;

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
    r13 = fmaf(r8, r13, r7);
  };
  LoadShared<4, float, float>(image_from_world,
                              4 * image_from_world_num_alloc,
                              image_from_world_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       image_from_world_indices_loc[threadIdx.x].target,
                       r7,
                       r15,
                       r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r10, r4, r13);
    r13 = fmaf(r9, r15, r13);
    r15 = copysign(1.0, r13);
    r15 = fmaf(r3, r15, r13);
    r15 = 1.0 / r15;
    r11 = fmaf(r8, r11, r5);
    r11 = fmaf(r9, r14, r11);
    r11 = fmaf(r10, r16, r11);
    r11 = fmaf(r11, r15, r0 * r2);
    r11 = r11 * r11;
    r0 = 2.00000000000000000e+00;
    r16 = 1.00000000000000000e+00;
  };
  LoadUnique<1, float, float>(reprojection_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r14 * r14;
    r14 = 1.0 / r14;
    r12 = fmaf(r8, r12, r6);
    r12 = fmaf(r10, r17, r12);
    r12 = fmaf(r9, r7, r12);
    r15 = fmaf(r12, r15, r1 * r2);
    r15 = r15 * r15;
    r12 = r11 + r15;
    r14 = fmaf(r12, r14, r16);
    r14 = sqrtf(r14);
    r14 = r16 + r14;
    r14 = 1.0 / r14;
    r14 = r0 * r14;
    r14 = fmaf(r15, r14, r11 * r14);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r14);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void FixedCameraPinholePointScore(float* point,
                                  unsigned int point_num_alloc,
                                  SharedIndex* point_indices,
                                  float* image_from_world,
                                  unsigned int image_from_world_num_alloc,
                                  SharedIndex* image_from_world_indices,
                                  float* pixel,
                                  unsigned int pixel_num_alloc,
                                  const float* const reprojection_loss_scale,
                                  float* const out_rTr,
                                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedCameraPinholePointScoreKernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      image_from_world,
      image_from_world_num_alloc,
      image_from_world_indices,
      pixel,
      pixel_num_alloc,
      reprojection_loss_scale,
      out_rTr,
      problem_size);
}

}  // namespace caspar