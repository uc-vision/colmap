#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rig_schur.h"
#include <cuda_runtime.h>

namespace caspar {
namespace {

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t size) : size_(size) {
    if (size_ > 0) {
      CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }
  }

  explicit DeviceBuffer(const std::vector<T>& values)
      : DeviceBuffer(values.size()) {
    if (size_ > 0) {
      CheckCuda(cudaMemcpy(
          data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  ~DeviceBuffer() { cudaFree(data_); }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

__device__ float LoadNodeValue(const float* values,
                               size_t node_num,
                               unsigned int node,
                               unsigned int dimension) {
  const size_t index = node;
  if (dimension < 4) {
    return values[4 * index + dimension];
  }
  return values[4 * node_num + 2 * index + dimension - 4];
}

__device__ void StoreNodeValue(float* values,
                               size_t node_num,
                               unsigned int node,
                               unsigned int dimension,
                               float value) {
  const size_t index = node;
  if (dimension < 4) {
    values[4 * index + dimension] = value;
  } else {
    values[4 * node_num + 2 * index + dimension - 4] = value;
  }
}

__device__ float LoadPoseTril(const float* values,
                              size_t pose_num,
                              unsigned int pose,
                              unsigned int slot) {
  return values[4 * pose_num * (slot / 4) + 4 * static_cast<size_t>(pose) +
                slot % 4];
}

__device__ unsigned int PoseTrilSlot(unsigned int row, unsigned int col) {
  unsigned int slot = 0;
  for (unsigned int slot_col = 0; slot_col < 6; ++slot_col) {
    for (unsigned int slot_row = slot_col + 1; slot_row < 6; ++slot_row) {
      if (slot_row == row && slot_col == col) {
        return slot;
      }
      ++slot;
    }
  }
  return slot;
}

__device__ float LoadPoseJac(const float* jac,
                             size_t factor_num,
                             size_t factor,
                             unsigned int row,
                             unsigned int col) {
  return jac[4 * factor_num * (col / 2) + 4 * factor + 2 * (col % 2) + row];
}

__device__ float LoadPointJac(const float* jac,
                              size_t factor_num,
                              size_t factor,
                              unsigned int row,
                              unsigned int col) {
  if (col < 2) {
    return jac[4 * factor + 2 * col + row];
  }
  return jac[4 * factor_num + 2 * factor + row];
}

__device__ float LoadScaleJac(const float* jac,
                              size_t factor,
                              unsigned int row) {
  return jac[2 * factor + row];
}

__global__ void FactorPointsKernel(const unsigned int* active_points,
                                   size_t active_point_num,
                                   size_t point_num,
                                   float lm_diag,
                                   const float* point_rhs,
                                   const float* point_diag,
                                   const float* point_tril,
                                   float* point_chol,
                                   float* point_y,
                                   int* info) {
  const size_t active_point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_point >= active_point_num) {
    return;
  }

  const unsigned int point = active_points[active_point];
  const float damping = 1.0f + lm_diag;
  const float diagonal_offset = lm_diag * 1e-6f;
  const size_t point_offset = 4 * static_cast<size_t>(point);
  const float h00 = fmaf(damping, point_diag[point_offset], diagonal_offset);
  const float h11 =
      fmaf(damping, point_diag[point_offset + 1], diagonal_offset);
  const float h22 =
      fmaf(damping, point_diag[point_offset + 2], diagonal_offset);
  const float h01 = point_tril[point_offset];
  const float h02 = point_tril[point_offset + 1];
  const float h12 = point_tril[point_offset + 2];

  if (!(h00 > 0.0f)) {
    atomicExch(info, 1);
    return;
  }
  const float l00 = sqrtf(h00);
  const float l10 = h01 / l00;
  const float l20 = h02 / l00;
  const float pivot1 = h11 - l10 * l10;
  if (!(pivot1 > 0.0f)) {
    atomicExch(info, 1);
    return;
  }
  const float l11 = sqrtf(pivot1);
  const float l21 = (h12 - l20 * l10) / l11;
  const float pivot2 = h22 - l20 * l20 - l21 * l21;
  if (!(pivot2 > 0.0f)) {
    atomicExch(info, 1);
    return;
  }
  const float l22 = sqrtf(pivot2);

  float* chol = point_chol + 6 * static_cast<size_t>(point);
  chol[0] = l00;
  chol[1] = l10;
  chol[2] = l11;
  chol[3] = l20;
  chol[4] = l21;
  chol[5] = l22;

  const float b0 = point_rhs[point_offset];
  const float b1 = point_rhs[point_offset + 1];
  const float b2 = point_rhs[point_offset + 2];
  const float y0 = b0 / l00;
  const float y1 = (b1 - l10 * y0) / l11;
  const float y2 = (b2 - l20 * y0 - l21 * y1) / l22;
  const size_t point_y_offset = 3 * static_cast<size_t>(point);
  point_y[point_y_offset] = y0;
  point_y[point_y_offset + 1] = y1;
  point_y[point_y_offset + 2] = y2;
}

__global__ void TransformEdgesKernel(const unsigned int* edge_points,
                                     const size_t* edge_factor_offsets,
                                     const size_t* edge_factor_indices,
                                     size_t edge_num,
                                     size_t factor_num,
                                     const float* pose_jac,
                                     const float* point_jac,
                                     const float* point_chol,
                                     float* edge_transform) {
  const size_t edge =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (edge >= edge_num) {
    return;
  }

  float cross[18] = {};
  for (size_t position = edge_factor_offsets[edge];
       position < edge_factor_offsets[edge + 1];
       ++position) {
    const size_t factor = edge_factor_indices[position];
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      for (unsigned int pose_dim = 0; pose_dim < 6; ++pose_dim) {
        float value = 0.0f;
        for (unsigned int row = 0; row < 2; ++row) {
          value += LoadPointJac(point_jac, factor_num, factor, row, point_dim) *
                   LoadPoseJac(pose_jac, factor_num, factor, row, pose_dim);
        }
        cross[6 * point_dim + pose_dim] += value;
      }
    }
  }

  const float* chol = point_chol + 6 * static_cast<size_t>(edge_points[edge]);
  for (unsigned int pose_dim = 0; pose_dim < 6; ++pose_dim) {
    const float t0 = cross[pose_dim] / chol[0];
    const float t1 = (cross[6 + pose_dim] - chol[1] * t0) / chol[2];
    const float t2 =
        (cross[12 + pose_dim] - chol[3] * t0 - chol[4] * t1) / chol[5];
    edge_transform[18 * edge + pose_dim] = t0;
    edge_transform[18 * edge + 6 + pose_dim] = t1;
    edge_transform[18 * edge + 12 + pose_dim] = t2;
  }
}

__global__ void TransformPointScaleKernel(const unsigned int* active_points,
                                          size_t active_point_num,
                                          const size_t* point_edge_offsets,
                                          const size_t* edge_factor_offsets,
                                          const size_t* edge_factor_indices,
                                          size_t factor_num,
                                          const float* scale_jac,
                                          const float* point_jac,
                                          const float* point_chol,
                                          float* point_scale_transform) {
  const size_t active_point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_point >= active_point_num) {
    return;
  }

  const unsigned int point = active_points[active_point];
  float cross[3] = {};
  for (size_t edge = point_edge_offsets[point];
       edge < point_edge_offsets[point + 1];
       ++edge) {
    for (size_t position = edge_factor_offsets[edge];
         position < edge_factor_offsets[edge + 1];
         ++position) {
      const size_t factor = edge_factor_indices[position];
      for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
        for (unsigned int row = 0; row < 2; ++row) {
          cross[point_dim] +=
              LoadPointJac(point_jac, factor_num, factor, row, point_dim) *
              LoadScaleJac(scale_jac, factor, row);
        }
      }
    }
  }

  const float* chol = point_chol + 6 * static_cast<size_t>(point);
  const float q0 = cross[0] / chol[0];
  const float q1 = (cross[1] - chol[1] * q0) / chol[2];
  const float q2 = (cross[2] - chol[3] * q0 - chol[4] * q1) / chol[5];
  point_scale_transform[3 * static_cast<size_t>(point)] = q0;
  point_scale_transform[3 * static_cast<size_t>(point) + 1] = q1;
  point_scale_transform[3 * static_cast<size_t>(point) + 2] = q2;
}

__global__ void InitializePoseSystemKernel(const unsigned int* active_poses,
                                           size_t active_pose_num,
                                           size_t pose_num,
                                           size_t rotation_anchor_pose,
                                           float lm_diag,
                                           const float* pose_rhs,
                                           const float* pose_diag,
                                           const float* pose_tril,
                                           float* pose_blocks,
                                           float* reduced_rhs) {
  const size_t compact_pose =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (compact_pose >= active_pose_num) {
    return;
  }

  const unsigned int pose = active_poses[compact_pose];
  for (unsigned int row = 0; row < 6; ++row) {
    reduced_rhs[6 * compact_pose + row] =
        compact_pose == rotation_anchor_pose && row < 3
            ? 0.0f
            : LoadNodeValue(pose_rhs, pose_num, pose, row);
    for (unsigned int col = 0; col < 6; ++col) {
      float value;
      if (row == col) {
        value = fmaf(1.0f + lm_diag,
                     LoadNodeValue(pose_diag, pose_num, pose, row),
                     lm_diag * 1e-6f);
      } else {
        const unsigned int lower_row = max(row, col);
        const unsigned int lower_col = min(row, col);
        value = LoadPoseTril(
            pose_tril, pose_num, pose, PoseTrilSlot(lower_row, lower_col));
      }
      pose_blocks[36 * compact_pose + 6 * row + col] = value;
    }
  }
}

__global__ void InitializeScaleSystemKernel(float lm_diag,
                                            const float* scale_rhs,
                                            const float* scale_diag,
                                            float* reduced_scale_rhs,
                                            float* reduced_scale_diagonal) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  reduced_scale_rhs[0] = scale_rhs[0];
  reduced_scale_diagonal[0] =
      fmaf(1.0f + lm_diag, scale_diag[0], lm_diag * 1e-6f);
}

__global__ void ReducePoseRhsKernel(const size_t* pose_edge_offsets,
                                    const size_t* pose_edges,
                                    size_t active_pose_num,
                                    const unsigned int* edge_points,
                                    const float* point_y,
                                    const float* edge_transform,
                                    size_t rotation_anchor_pose,
                                    float* reduced_rhs) {
  const size_t pose = blockIdx.x;
  if (pose >= active_pose_num) {
    return;
  }

  const unsigned int pose_dim = threadIdx.x / warpSize;
  const unsigned int lane = threadIdx.x % warpSize;
  float correction = 0.0f;
  for (size_t position = pose_edge_offsets[pose] + lane;
       position < pose_edge_offsets[pose + 1];
       position += warpSize) {
    const size_t edge = pose_edges[position];
    const unsigned int point = edge_points[edge];
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      correction += edge_transform[18 * edge + 6 * point_dim + pose_dim] *
                    point_y[3 * static_cast<size_t>(point) + point_dim];
    }
  }
  for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
    correction += __shfl_down_sync(0xffffffff, correction, offset);
  }
  if (lane == 0) {
    reduced_rhs[6 * pose + pose_dim] =
        pose == rotation_anchor_pose && pose_dim < 3
            ? 0.0f
            : reduced_rhs[6 * pose + pose_dim] - correction;
  }
}

__global__ void FactorPosePreconditionerKernel(const size_t* pose_edge_offsets,
                                               const size_t* pose_edges,
                                               size_t active_pose_num,
                                               size_t rotation_anchor_pose,
                                               const float* edge_transform,
                                               const float* pose_blocks,
                                               float* pose_preconditioner_chol,
                                               int* info) {
  const size_t pose = blockIdx.x;
  const unsigned int element = threadIdx.x;
  if (pose >= active_pose_num || element >= 64) {
    return;
  }

  __shared__ float block[36];
  __shared__ int valid;
  if (element < 36) {
    const unsigned int row = element / 6;
    const unsigned int col = element % 6;
    float value = pose_blocks[36 * pose + element];
    for (size_t position = pose_edge_offsets[pose];
         position < pose_edge_offsets[pose + 1];
         ++position) {
      const size_t edge = pose_edges[position];
      for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
        value -= edge_transform[18 * edge + 6 * point_dim + row] *
                 edge_transform[18 * edge + 6 * point_dim + col];
      }
    }
    block[element] = value;
  }
  __syncthreads();

  if (element == 0) {
    valid = 1;
    if (pose == rotation_anchor_pose) {
      for (unsigned int row = 0; row < 6; ++row) {
        for (unsigned int col = 0; col < 6; ++col) {
          if (row < 3 || col < 3) {
            block[6 * row + col] = row == col && row < 3 ? 1.0f : 0.0f;
          }
        }
      }
    }
    for (unsigned int col = 0; col < 6; ++col) {
      float pivot = block[6 * col + col];
      for (unsigned int inner = 0; inner < col; ++inner) {
        pivot -= block[6 * col + inner] * block[6 * col + inner];
      }
      if (!(pivot > 0.0f)) {
        valid = 0;
        atomicExch(info, 1);
        break;
      }
      block[6 * col + col] = sqrtf(pivot);
      for (unsigned int row = col + 1; row < 6; ++row) {
        float value = block[6 * row + col];
        for (unsigned int inner = 0; inner < col; ++inner) {
          value -= block[6 * row + inner] * block[6 * col + inner];
        }
        block[6 * row + col] = value / block[6 * col + col];
      }
    }
  }
  __syncthreads();

  if (valid != 0 && element < 36) {
    pose_preconditioner_chol[36 * pose + element] = block[element];
  }
}

__global__ void ReduceScaleRhsDiagonalKernel(const unsigned int* active_points,
                                             size_t active_point_num,
                                             const float* point_y,
                                             const float* point_scale_transform,
                                             float* reduced_scale_rhs,
                                             float* reduced_scale_diagonal) {
  const size_t active_point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float rhs_correction = 0.0f;
  float diagonal_correction = 0.0f;
  if (active_point < active_point_num) {
    const unsigned int point = active_points[active_point];
    const size_t point_offset = 3 * static_cast<size_t>(point);
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      const float q = point_scale_transform[point_offset + point_dim];
      rhs_correction =
          fmaf(q, point_y[point_offset + point_dim], rhs_correction);
      diagonal_correction = fmaf(q, q, diagonal_correction);
    }
  }
  __shared__ float rhs_block[256];
  __shared__ float diagonal_block[256];
  rhs_block[threadIdx.x] = rhs_correction;
  diagonal_block[threadIdx.x] = diagonal_correction;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      rhs_block[threadIdx.x] += rhs_block[threadIdx.x + offset];
      diagonal_block[threadIdx.x] += diagonal_block[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(reduced_scale_rhs, -rhs_block[0]);
    atomicAdd(reduced_scale_diagonal, -diagonal_block[0]);
  }
}

__global__ void ReducePoseScaleKernel(const size_t* pose_edge_offsets,
                                      const size_t* pose_edges,
                                      const size_t* edge_factor_offsets,
                                      const size_t* edge_factor_indices,
                                      size_t active_pose_num,
                                      size_t factor_num,
                                      const unsigned int* edge_points,
                                      const float* pose_jac,
                                      const float* scale_jac,
                                      const float* edge_transform,
                                      const float* point_scale_transform,
                                      size_t rotation_anchor_pose,
                                      float* reduced_pose_scale) {
  const size_t pose = blockIdx.x;
  if (pose >= active_pose_num) {
    return;
  }

  const unsigned int pose_dim = threadIdx.x / warpSize;
  const unsigned int lane = threadIdx.x % warpSize;
  float value = 0.0f;
  for (size_t position = pose_edge_offsets[pose] + lane;
       position < pose_edge_offsets[pose + 1];
       position += warpSize) {
    const size_t edge = pose_edges[position];
    for (size_t factor_position = edge_factor_offsets[edge];
         factor_position < edge_factor_offsets[edge + 1];
         ++factor_position) {
      const size_t factor = edge_factor_indices[factor_position];
      for (unsigned int row = 0; row < 2; ++row) {
        value += LoadPoseJac(pose_jac, factor_num, factor, row, pose_dim) *
                 LoadScaleJac(scale_jac, factor, row);
      }
    }
    const size_t point_offset = 3 * static_cast<size_t>(edge_points[edge]);
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      value -= edge_transform[18 * edge + 6 * point_dim + pose_dim] *
               point_scale_transform[point_offset + point_dim];
    }
  }
  for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) {
    reduced_pose_scale[6 * pose + pose_dim] =
        pose == rotation_anchor_pose && pose_dim < 3 ? 0.0f : value;
  }
}

__global__ void ProjectPointsKernel(const unsigned int* active_points,
                                    size_t active_point_num,
                                    const size_t* point_edge_offsets,
                                    const size_t* edge_poses,
                                    const float* edge_transform,
                                    const float* vector,
                                    float* point_projection) {
  const size_t active_point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_point >= active_point_num) {
    return;
  }

  const unsigned int point = active_points[active_point];
  float projection[3] = {};
  for (size_t edge = point_edge_offsets[point];
       edge < point_edge_offsets[point + 1];
       ++edge) {
    const size_t pose = edge_poses[edge];
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      for (unsigned int pose_dim = 0; pose_dim < 6; ++pose_dim) {
        projection[point_dim] +=
            edge_transform[18 * edge + 6 * point_dim + pose_dim] *
            vector[6 * pose + pose_dim];
      }
    }
  }
  for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
    point_projection[3 * static_cast<size_t>(point) + point_dim] =
        projection[point_dim];
  }
}

__global__ void PoseMatVecKernel(const size_t* pose_edge_offsets,
                                 const size_t* pose_edges,
                                 size_t active_pose_num,
                                 size_t rotation_anchor_pose,
                                 const unsigned int* edge_points,
                                 const float* edge_transform,
                                 const float* point_projection,
                                 const float* pose_blocks,
                                 const float* vector,
                                 float* product) {
  const size_t pose = blockIdx.x;
  if (pose >= active_pose_num) {
    return;
  }

  const unsigned int pose_dim = threadIdx.x / warpSize;
  const unsigned int lane = threadIdx.x % warpSize;
  float value = 0.0f;
  if (lane == 0) {
    for (unsigned int col = 0; col < 6; ++col) {
      value +=
          pose_blocks[36 * pose + 6 * pose_dim + col] * vector[6 * pose + col];
    }
  }
  for (size_t position = pose_edge_offsets[pose] + lane;
       position < pose_edge_offsets[pose + 1];
       position += warpSize) {
    const size_t edge = pose_edges[position];
    const size_t point_offset = 3 * static_cast<size_t>(edge_points[edge]);
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      value -= edge_transform[18 * edge + 6 * point_dim + pose_dim] *
               point_projection[point_offset + point_dim];
    }
  }
  for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) {
    product[6 * pose + pose_dim] = pose == rotation_anchor_pose && pose_dim < 3
                                       ? vector[6 * pose + pose_dim]
                                       : value;
  }
}

__global__ void ApplyPosePreconditionerKernel(
    size_t active_pose_num,
    const float* pose_preconditioner_chol,
    const float* residual,
    float* preconditioned) {
  const size_t pose =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pose >= active_pose_num) {
    return;
  }

  const float* chol = pose_preconditioner_chol + 36 * pose;
  float value[6];
  for (unsigned int row = 0; row < 6; ++row) {
    value[row] = residual[6 * pose + row];
    for (unsigned int col = 0; col < row; ++col) {
      value[row] -= chol[6 * row + col] * value[col];
    }
    value[row] /= chol[6 * row + row];
  }
  for (int row = 5; row >= 0; --row) {
    for (unsigned int col = row + 1; col < 6; ++col) {
      value[row] -= chol[6 * col + row] * value[col];
    }
    value[row] /= chol[6 * row + row];
  }
  for (unsigned int row = 0; row < 6; ++row) {
    preconditioned[6 * pose + row] = value[row];
  }
}

__global__ void UpdatePcgSolutionResidualKernel(size_t dimension,
                                                float alpha,
                                                const float* direction,
                                                const float* product,
                                                float* solution,
                                                float* residual) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < dimension) {
    solution[index] = fmaf(alpha, direction[index], solution[index]);
    residual[index] = fmaf(-alpha, product[index], residual[index]);
  }
}

__global__ void UpdatePcgDirectionKernel(size_t dimension,
                                         float beta,
                                         const float* preconditioned,
                                         float* direction) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < dimension) {
    direction[index] = fmaf(beta, direction[index], preconditioned[index]);
  }
}

__global__ void DotKernel(size_t dimension,
                          const float* lhs,
                          const float* rhs,
                          float* result) {
  float value = 0.0f;
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < dimension;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    value = fmaf(lhs[index], rhs[index], value);
  }
  for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  __shared__ float warp_sums[8];
  const unsigned int lane = threadIdx.x % warpSize;
  const unsigned int warp = threadIdx.x / warpSize;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < blockDim.x / warpSize ? warp_sums[lane] : 0.0f;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      atomicAdd(result, value);
    }
  }
}

__global__ void CombineArrowheadStepKernel(size_t pose_dimension,
                                           float scale_step,
                                           const float* pose_solution,
                                           const float* scale_response,
                                           float* reduced_step) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pose_dimension) {
    reduced_step[index] =
        fmaf(-scale_step, scale_response[index], pose_solution[index]);
  }
  if (index == 0) {
    reduced_step[pose_dimension] = scale_step;
  }
}

__global__ void ScatterPoseStepKernel(const unsigned int* active_poses,
                                      size_t active_pose_num,
                                      size_t pose_num,
                                      const float* reduced_step,
                                      float* pose_step) {
  const size_t compact_pose =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (compact_pose >= active_pose_num) {
    return;
  }
  const unsigned int pose = active_poses[compact_pose];
  for (unsigned int dimension = 0; dimension < 6; ++dimension) {
    StoreNodeValue(pose_step,
                   pose_num,
                   pose,
                   dimension,
                   reduced_step[6 * compact_pose + dimension]);
  }
}

__global__ void BackSubstitutePointsKernel(const unsigned int* active_points,
                                           size_t active_point_num,
                                           size_t point_num,
                                           const size_t* point_edge_offsets,
                                           const size_t* edge_poses,
                                           const float* point_chol,
                                           const float* point_y,
                                           const float* edge_transform,
                                           const float* point_scale_transform,
                                           const float* reduced_step,
                                           size_t pose_dimension,
                                           float* point_step) {
  const size_t active_point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_point >= active_point_num) {
    return;
  }

  const unsigned int point = active_points[active_point];
  const size_t point_y_offset = 3 * static_cast<size_t>(point);
  float value[3] = {point_y[point_y_offset],
                    point_y[point_y_offset + 1],
                    point_y[point_y_offset + 2]};
  for (size_t edge = point_edge_offsets[point];
       edge < point_edge_offsets[point + 1];
       ++edge) {
    const size_t pose = edge_poses[edge];
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      for (unsigned int pose_dim = 0; pose_dim < 6; ++pose_dim) {
        value[point_dim] -=
            edge_transform[18 * edge + 6 * point_dim + pose_dim] *
            reduced_step[6 * pose + pose_dim];
      }
    }
  }
  if (point_scale_transform != nullptr) {
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      value[point_dim] -= point_scale_transform[point_y_offset + point_dim] *
                          reduced_step[pose_dimension];
    }
  }

  const float* chol = point_chol + 6 * static_cast<size_t>(point);
  const float x2 = value[2] / chol[5];
  const float x1 = (value[1] - chol[4] * x2) / chol[2];
  const float x0 = (value[0] - chol[1] * x1 - chol[3] * x2) / chol[0];
  const size_t point_offset = 4 * static_cast<size_t>(point);
  point_step[point_offset] = x0;
  point_step[point_offset + 1] = x1;
  point_step[point_offset + 2] = x2;
}

}  // namespace

class RigSchurSolver::Impl {
 public:
  Impl(int device_id,
       size_t pose_num,
       size_t point_num,
       const std::vector<unsigned int>& pose_indices,
       const std::vector<unsigned int>& point_indices,
       const std::vector<unsigned int>& fixed_pose_point_indices,
       const std::vector<unsigned int>& fixed_point_pose_indices,
       bool scale_aware,
       int rotation_anchor_pose_index)
      : device_id_(device_id),
        pose_num_(pose_num),
        point_num_(point_num),
        factor_num_(pose_indices.size()),
        scale_aware_(scale_aware) {
    CheckCuda(cudaSetDevice(device_id_));
    cudaDeviceProp device_properties;
    CheckCuda(cudaGetDeviceProperties(&device_properties, device_id_));
    max_grid_size_ = static_cast<size_t>(device_properties.maxGridSize[0]);
    if (pose_indices.size() != point_indices.size()) {
      throw std::invalid_argument("Rig Schur pose/point index size mismatch");
    }
    if (pose_indices.empty() || pose_num == 0 || point_num == 0) {
      throw std::invalid_argument(
          "Rig Schur requires poses, points, and factors");
    }
    const size_t max_generated_count = std::numeric_limits<unsigned int>::max();
    if (pose_num > max_generated_count || point_num > max_generated_count ||
        factor_num_ > max_generated_count) {
      throw std::invalid_argument("Rig Schur generated topology is too large");
    }

    std::vector<size_t> point_factor_offsets(point_num + 1, 0);
    std::vector<char> active_pose_flags(pose_num, false);
    std::vector<char> active_point_flags(point_num, false);
    for (size_t factor = 0; factor < pose_indices.size(); ++factor) {
      if (pose_indices[factor] >= pose_num ||
          point_indices[factor] >= point_num) {
        throw std::out_of_range("Rig Schur factor index out of range");
      }
      ++point_factor_offsets[point_indices[factor] + 1];
      active_pose_flags[pose_indices[factor]] = true;
      active_point_flags[point_indices[factor]] = true;
    }
    for (const unsigned int point : fixed_pose_point_indices) {
      if (point >= point_num) {
        throw std::out_of_range(
            "Rig Schur fixed-pose point index out of range");
      }
      active_point_flags[point] = true;
    }
    for (const unsigned int pose : fixed_point_pose_indices) {
      if (pose >= pose_num) {
        throw std::out_of_range(
            "Rig Schur fixed-point pose index out of range");
      }
      active_pose_flags[pose] = true;
    }
    std::partial_sum(point_factor_offsets.begin(),
                     point_factor_offsets.end(),
                     point_factor_offsets.begin());

    std::vector<size_t> factor_indices(pose_indices.size());
    std::vector<size_t> point_factor_cursors = point_factor_offsets;
    for (size_t factor = 0; factor < factor_num_; ++factor) {
      factor_indices[point_factor_cursors[point_indices[factor]]++] = factor;
    }
    for (size_t point = 0; point < point_num_; ++point) {
      const auto begin = factor_indices.begin() + point_factor_offsets[point];
      const auto end = factor_indices.begin() + point_factor_offsets[point + 1];
      std::sort(begin, end, [&](size_t lhs, size_t rhs) {
        return pose_indices[lhs] < pose_indices[rhs];
      });
    }

    std::vector<unsigned int> active_poses;
    std::vector<size_t> pose_to_compact(pose_num,
                                        std::numeric_limits<size_t>::max());
    for (size_t pose = 0; pose < pose_num_; ++pose) {
      if (active_pose_flags[pose]) {
        pose_to_compact[pose] = active_poses.size();
        active_poses.push_back(static_cast<unsigned int>(pose));
      }
    }
    if (rotation_anchor_pose_index >= 0) {
      if (static_cast<size_t>(rotation_anchor_pose_index) >= pose_num_ ||
          pose_to_compact[rotation_anchor_pose_index] ==
              std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument(
            "Rig Schur rotation anchor is not an active pose");
      }
      rotation_anchor_compact_pose_ =
          pose_to_compact[rotation_anchor_pose_index];
    }
    std::vector<unsigned int> active_points;
    for (size_t point = 0; point < point_num_; ++point) {
      if (active_point_flags[point]) {
        active_points.push_back(static_cast<unsigned int>(point));
      }
    }
    active_pose_num_ = active_poses.size();
    active_point_num_ = active_points.size();
    if (active_pose_num_ >
        (static_cast<size_t>(std::numeric_limits<int>::max()) -
         static_cast<size_t>(scale_aware_)) /
            6) {
      throw std::invalid_argument("Rig Schur reduced system is too large");
    }
    pose_dimension_ = 6 * active_pose_num_;

    std::vector<unsigned int> edge_points;
    std::vector<size_t> edge_poses;
    std::vector<size_t> edge_factor_offsets{0};
    std::vector<size_t> edge_factor_indices;
    std::vector<size_t> point_edge_offsets(point_num + 1);
    edge_factor_indices.reserve(factor_num_);
    for (size_t point = 0; point < point_num_; ++point) {
      point_edge_offsets[point] = edge_points.size();
      size_t position = point_factor_offsets[point];
      const size_t end = point_factor_offsets[point + 1];
      while (position < end) {
        const unsigned int pose = pose_indices[factor_indices[position]];
        edge_points.push_back(static_cast<unsigned int>(point));
        edge_poses.push_back(pose_to_compact[pose]);
        do {
          edge_factor_indices.push_back(factor_indices[position]);
          ++position;
        } while (position < end &&
                 pose_indices[factor_indices[position]] == pose);
        edge_factor_offsets.push_back(edge_factor_indices.size());
      }
    }
    point_edge_offsets[point_num_] = edge_points.size();
    edge_num_ = edge_points.size();

    std::vector<size_t> pose_edge_offsets(active_pose_num_ + 1, 0);
    for (const size_t pose : edge_poses) {
      ++pose_edge_offsets[pose + 1];
    }
    std::partial_sum(pose_edge_offsets.begin(),
                     pose_edge_offsets.end(),
                     pose_edge_offsets.begin());
    std::vector<size_t> pose_edges(edge_num_);
    std::vector<size_t> pose_edge_cursors = pose_edge_offsets;
    for (size_t edge = 0; edge < edge_num_; ++edge) {
      pose_edges[pose_edge_cursors[edge_poses[edge]]++] = edge;
    }

    active_poses_ = DeviceBuffer<unsigned int>(active_poses);
    active_points_ = DeviceBuffer<unsigned int>(active_points);
    edge_points_ = DeviceBuffer<unsigned int>(edge_points);
    edge_poses_ = DeviceBuffer<size_t>(edge_poses);
    edge_factor_offsets_ = DeviceBuffer<size_t>(edge_factor_offsets);
    edge_factor_indices_ = DeviceBuffer<size_t>(edge_factor_indices);
    point_edge_offsets_ = DeviceBuffer<size_t>(point_edge_offsets);
    pose_edge_offsets_ = DeviceBuffer<size_t>(pose_edge_offsets);
    pose_edges_ = DeviceBuffer<size_t>(pose_edges);
    point_chol_ = DeviceBuffer<float>(6 * point_num_);
    point_y_ = DeviceBuffer<float>(3 * point_num_);
    edge_transform_ = DeviceBuffer<float>(18 * edge_num_);
    point_scale_transform_ =
        DeviceBuffer<float>(scale_aware_ ? 3 * point_num_ : 0);
    point_projection_ = DeviceBuffer<float>(3 * point_num_);
    pose_blocks_ = DeviceBuffer<float>(36 * active_pose_num_);
    pose_preconditioner_chol_ = DeviceBuffer<float>(36 * active_pose_num_);
    reduced_rhs_ = DeviceBuffer<float>(pose_dimension_);
    reduced_pose_scale_ =
        DeviceBuffer<float>(scale_aware_ ? pose_dimension_ : 0);
    reduced_scale_rhs_ = DeviceBuffer<float>(scale_aware_ ? 1 : 0);
    reduced_scale_diagonal_ = DeviceBuffer<float>(scale_aware_ ? 1 : 0);
    reduced_step_ = DeviceBuffer<float>(pose_dimension_ +
                                        static_cast<size_t>(scale_aware_));
    scale_pose_response_ =
        DeviceBuffer<float>(scale_aware_ ? pose_dimension_ : 0);
    pcg_residual_ = DeviceBuffer<float>(pose_dimension_);
    pcg_preconditioned_ = DeviceBuffer<float>(pose_dimension_);
    pcg_direction_ = DeviceBuffer<float>(pose_dimension_);
    pcg_product_ = DeviceBuffer<float>(pose_dimension_);
    dot_result_ = DeviceBuffer<float>(1);
    info_ = DeviceBuffer<int>(1);
  }

  ~Impl() { cudaSetDevice(device_id_); }

  unsigned int GridSize(size_t count, unsigned int block_size) const {
    const size_t grid_size =
        count / block_size + static_cast<size_t>(count % block_size != 0);
    if (grid_size > max_grid_size_) {
      throw std::invalid_argument("Rig Schur CUDA grid is too large");
    }
    return static_cast<unsigned int>(grid_size);
  }

  float Dot(const float* lhs, const float* rhs) {
    constexpr unsigned int block_size = 256;
    CheckCuda(cudaMemset(dot_result_.data(), 0, sizeof(float)));
    DotKernel<<<GridSize(pose_dimension_, block_size), block_size>>>(
        pose_dimension_, lhs, rhs, dot_result_.data());
    CheckCuda(cudaGetLastError());
    float result;
    CheckCuda(cudaMemcpy(
        &result, dot_result_.data(), sizeof(float), cudaMemcpyDeviceToHost));
    return result;
  }

  void PoseMatVec(const float* vector, float* product) {
    constexpr unsigned int block_size = 256;
    ProjectPointsKernel<<<GridSize(active_point_num_, block_size),
                          block_size>>>(active_points_.data(),
                                        active_point_num_,
                                        point_edge_offsets_.data(),
                                        edge_poses_.data(),
                                        edge_transform_.data(),
                                        vector,
                                        point_projection_.data());
    CheckCuda(cudaGetLastError());
    PoseMatVecKernel<<<GridSize(active_pose_num_, 1), 6 * 32>>>(
        pose_edge_offsets_.data(),
        pose_edges_.data(),
        active_pose_num_,
        rotation_anchor_compact_pose_,
        edge_points_.data(),
        edge_transform_.data(),
        point_projection_.data(),
        pose_blocks_.data(),
        vector,
        product);
    CheckCuda(cudaGetLastError());
  }

  bool SolvePoseSystem(const float* rhs,
                       int iteration_max,
                       float relative_error_exit,
                       float* solution) {
    constexpr unsigned int block_size = 256;
    const size_t bytes = pose_dimension_ * sizeof(float);
    CheckCuda(cudaMemset(solution, 0, bytes));
    CheckCuda(
        cudaMemcpy(pcg_residual_.data(), rhs, bytes, cudaMemcpyDeviceToDevice));
    const float initial_residual_norm =
        Dot(pcg_residual_.data(), pcg_residual_.data());
    if (initial_residual_norm == 0.0f) {
      return true;
    }
    ApplyPosePreconditionerKernel<<<GridSize(active_pose_num_, block_size),
                                    block_size>>>(
        active_pose_num_,
        pose_preconditioner_chol_.data(),
        pcg_residual_.data(),
        pcg_preconditioned_.data());
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaMemcpy(pcg_direction_.data(),
                         pcg_preconditioned_.data(),
                         bytes,
                         cudaMemcpyDeviceToDevice));
    float residual_preconditioned =
        Dot(pcg_residual_.data(), pcg_preconditioned_.data());
    if (!(residual_preconditioned > 0.0f)) {
      return false;
    }

    for (int iteration = 0; iteration < iteration_max; ++iteration) {
      PoseMatVec(pcg_direction_.data(), pcg_product_.data());
      const float direction_product =
          Dot(pcg_direction_.data(), pcg_product_.data());
      if (!(direction_product > 0.0f)) {
        return false;
      }
      const float alpha = residual_preconditioned / direction_product;
      UpdatePcgSolutionResidualKernel<<<GridSize(pose_dimension_, block_size),
                                        block_size>>>(pose_dimension_,
                                                      alpha,
                                                      pcg_direction_.data(),
                                                      pcg_product_.data(),
                                                      solution,
                                                      pcg_residual_.data());
      CheckCuda(cudaGetLastError());
      const float residual_norm =
          Dot(pcg_residual_.data(), pcg_residual_.data());
      if (residual_norm <= initial_residual_norm * relative_error_exit) {
        return true;
      }
      ApplyPosePreconditionerKernel<<<GridSize(active_pose_num_, block_size),
                                      block_size>>>(
          active_pose_num_,
          pose_preconditioner_chol_.data(),
          pcg_residual_.data(),
          pcg_preconditioned_.data());
      CheckCuda(cudaGetLastError());
      const float next_residual_preconditioned =
          Dot(pcg_residual_.data(), pcg_preconditioned_.data());
      if (!(next_residual_preconditioned > 0.0f)) {
        return false;
      }
      const float beta = next_residual_preconditioned / residual_preconditioned;
      UpdatePcgDirectionKernel<<<GridSize(pose_dimension_, block_size),
                                 block_size>>>(pose_dimension_,
                                               beta,
                                               pcg_preconditioned_.data(),
                                               pcg_direction_.data());
      CheckCuda(cudaGetLastError());
      residual_preconditioned = next_residual_preconditioned;
    }
    return true;
  }

  bool SolveStep(float diag,
                 int pcg_iteration_max,
                 float pcg_relative_error_exit,
                 const float* pose_jac,
                 const float* scale_jac,
                 const float* point_jac,
                 const float* pose_rhs,
                 const float* pose_diag,
                 const float* pose_tril,
                 const float* scale_rhs,
                 const float* scale_diag,
                 const float* point_rhs,
                 const float* point_diag,
                 const float* point_tril,
                 float* pose_step,
                 float* scale_step,
                 float* point_step) {
    CheckCuda(cudaSetDevice(device_id_));
    CheckCuda(cudaMemset(info_.data(), 0, sizeof(int)));
    constexpr unsigned int block_size = 256;
    FactorPointsKernel<<<GridSize(active_point_num_, block_size), block_size>>>(
        active_points_.data(),
        active_point_num_,
        point_num_,
        diag,
        point_rhs,
        point_diag,
        point_tril,
        point_chol_.data(),
        point_y_.data(),
        info_.data());
    CheckCuda(cudaGetLastError());
    int info = 0;
    CheckCuda(
        cudaMemcpy(&info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
      return false;
    }

    TransformEdgesKernel<<<GridSize(edge_num_, block_size), block_size>>>(
        edge_points_.data(),
        edge_factor_offsets_.data(),
        edge_factor_indices_.data(),
        edge_num_,
        factor_num_,
        pose_jac,
        point_jac,
        point_chol_.data(),
        edge_transform_.data());
    CheckCuda(cudaGetLastError());
    if (scale_aware_) {
      TransformPointScaleKernel<<<GridSize(active_point_num_, block_size),
                                  block_size>>>(active_points_.data(),
                                                active_point_num_,
                                                point_edge_offsets_.data(),
                                                edge_factor_offsets_.data(),
                                                edge_factor_indices_.data(),
                                                factor_num_,
                                                scale_jac,
                                                point_jac,
                                                point_chol_.data(),
                                                point_scale_transform_.data());
      CheckCuda(cudaGetLastError());
    }

    InitializePoseSystemKernel<<<GridSize(active_pose_num_, block_size),
                                 block_size>>>(active_poses_.data(),
                                               active_pose_num_,
                                               pose_num_,
                                               rotation_anchor_compact_pose_,
                                               diag,
                                               pose_rhs,
                                               pose_diag,
                                               pose_tril,
                                               pose_blocks_.data(),
                                               reduced_rhs_.data());
    CheckCuda(cudaGetLastError());
    if (scale_aware_) {
      InitializeScaleSystemKernel<<<1, 1>>>(diag,
                                            scale_rhs,
                                            scale_diag,
                                            reduced_scale_rhs_.data(),
                                            reduced_scale_diagonal_.data());
      CheckCuda(cudaGetLastError());
    }
    ReducePoseRhsKernel<<<GridSize(active_pose_num_, 1), 6 * 32>>>(
        pose_edge_offsets_.data(),
        pose_edges_.data(),
        active_pose_num_,
        edge_points_.data(),
        point_y_.data(),
        edge_transform_.data(),
        rotation_anchor_compact_pose_,
        reduced_rhs_.data());
    CheckCuda(cudaGetLastError());
    FactorPosePreconditionerKernel<<<GridSize(active_pose_num_, 1), 64>>>(
        pose_edge_offsets_.data(),
        pose_edges_.data(),
        active_pose_num_,
        rotation_anchor_compact_pose_,
        edge_transform_.data(),
        pose_blocks_.data(),
        pose_preconditioner_chol_.data(),
        info_.data());
    CheckCuda(cudaGetLastError());
    CheckCuda(
        cudaMemcpy(&info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
      return false;
    }
    if (scale_aware_) {
      ReduceScaleRhsDiagonalKernel<<<GridSize(active_point_num_, block_size),
                                     block_size>>>(
          active_points_.data(),
          active_point_num_,
          point_y_.data(),
          point_scale_transform_.data(),
          reduced_scale_rhs_.data(),
          reduced_scale_diagonal_.data());
      CheckCuda(cudaGetLastError());
      ReducePoseScaleKernel<<<GridSize(active_pose_num_, 1), 6 * 32>>>(
          pose_edge_offsets_.data(),
          pose_edges_.data(),
          edge_factor_offsets_.data(),
          edge_factor_indices_.data(),
          active_pose_num_,
          factor_num_,
          edge_points_.data(),
          pose_jac,
          scale_jac,
          edge_transform_.data(),
          point_scale_transform_.data(),
          rotation_anchor_compact_pose_,
          reduced_pose_scale_.data());
      CheckCuda(cudaGetLastError());
    }

    if (!SolvePoseSystem(reduced_rhs_.data(),
                         pcg_iteration_max,
                         pcg_relative_error_exit,
                         reduced_step_.data())) {
      return false;
    }
    if (scale_aware_) {
      if (!SolvePoseSystem(reduced_pose_scale_.data(),
                           pcg_iteration_max,
                           pcg_relative_error_exit,
                           scale_pose_response_.data())) {
        return false;
      }
      float reduced_scale_rhs;
      float reduced_scale_diagonal;
      CheckCuda(cudaMemcpy(&reduced_scale_rhs,
                           reduced_scale_rhs_.data(),
                           sizeof(float),
                           cudaMemcpyDeviceToHost));
      CheckCuda(cudaMemcpy(&reduced_scale_diagonal,
                           reduced_scale_diagonal_.data(),
                           sizeof(float),
                           cudaMemcpyDeviceToHost));
      const float pose_scale_solution =
          Dot(reduced_pose_scale_.data(), reduced_step_.data());
      const float pose_scale_response =
          Dot(reduced_pose_scale_.data(), scale_pose_response_.data());
      const float scale_denominator =
          reduced_scale_diagonal - pose_scale_response;
      if (!(scale_denominator > 0.0f)) {
        return false;
      }
      const float solved_scale =
          (reduced_scale_rhs - pose_scale_solution) / scale_denominator;
      CombineArrowheadStepKernel<<<GridSize(pose_dimension_, block_size),
                                   block_size>>>(pose_dimension_,
                                                 solved_scale,
                                                 reduced_step_.data(),
                                                 scale_pose_response_.data(),
                                                 reduced_step_.data());
      CheckCuda(cudaGetLastError());
    }

    ScatterPoseStepKernel<<<GridSize(active_pose_num_, block_size),
                            block_size>>>(active_poses_.data(),
                                          active_pose_num_,
                                          pose_num_,
                                          reduced_step_.data(),
                                          pose_step);
    CheckCuda(cudaGetLastError());
    if (scale_aware_) {
      CheckCuda(cudaMemcpy(scale_step,
                           reduced_step_.data() + pose_dimension_,
                           sizeof(float),
                           cudaMemcpyDeviceToDevice));
    }
    BackSubstitutePointsKernel<<<GridSize(active_point_num_, block_size),
                                 block_size>>>(
        active_points_.data(),
        active_point_num_,
        point_num_,
        point_edge_offsets_.data(),
        edge_poses_.data(),
        point_chol_.data(),
        point_y_.data(),
        edge_transform_.data(),
        scale_aware_ ? point_scale_transform_.data() : nullptr,
        reduced_step_.data(),
        pose_dimension_,
        point_step);
    CheckCuda(cudaGetLastError());
    return true;
  }

 private:
  int device_id_;
  size_t pose_num_;
  size_t point_num_;
  size_t factor_num_;
  bool scale_aware_;
  size_t active_pose_num_;
  size_t active_point_num_;
  size_t edge_num_;
  size_t max_grid_size_;
  size_t pose_dimension_;
  size_t rotation_anchor_compact_pose_ = std::numeric_limits<size_t>::max();
  DeviceBuffer<unsigned int> active_poses_;
  DeviceBuffer<unsigned int> active_points_;
  DeviceBuffer<unsigned int> edge_points_;
  DeviceBuffer<size_t> edge_poses_;
  DeviceBuffer<size_t> edge_factor_offsets_;
  DeviceBuffer<size_t> edge_factor_indices_;
  DeviceBuffer<size_t> point_edge_offsets_;
  DeviceBuffer<size_t> pose_edge_offsets_;
  DeviceBuffer<size_t> pose_edges_;
  DeviceBuffer<float> point_chol_;
  DeviceBuffer<float> point_y_;
  DeviceBuffer<float> edge_transform_;
  DeviceBuffer<float> point_scale_transform_;
  DeviceBuffer<float> point_projection_;
  DeviceBuffer<float> pose_blocks_;
  DeviceBuffer<float> pose_preconditioner_chol_;
  DeviceBuffer<float> reduced_rhs_;
  DeviceBuffer<float> reduced_pose_scale_;
  DeviceBuffer<float> reduced_scale_rhs_;
  DeviceBuffer<float> reduced_scale_diagonal_;
  DeviceBuffer<float> reduced_step_;
  DeviceBuffer<float> scale_pose_response_;
  DeviceBuffer<float> pcg_residual_;
  DeviceBuffer<float> pcg_preconditioned_;
  DeviceBuffer<float> pcg_direction_;
  DeviceBuffer<float> pcg_product_;
  DeviceBuffer<float> dot_result_;
  DeviceBuffer<int> info_;
};

RigSchurSolver::RigSchurSolver(
    int device_id,
    size_t pose_num,
    size_t point_num,
    const std::vector<unsigned int>& pose_indices,
    const std::vector<unsigned int>& point_indices,
    const std::vector<unsigned int>& fixed_pose_point_indices,
    const std::vector<unsigned int>& fixed_point_pose_indices,
    bool scale_aware,
    int rotation_anchor_pose_index)
    : impl_(std::make_unique<Impl>(device_id,
                                   pose_num,
                                   point_num,
                                   pose_indices,
                                   point_indices,
                                   fixed_pose_point_indices,
                                   fixed_point_pose_indices,
                                   scale_aware,
                                   rotation_anchor_pose_index)) {}

RigSchurSolver::~RigSchurSolver() = default;

bool RigSchurSolver::SolveStep(float diag,
                               int pcg_iteration_max,
                               float pcg_relative_error_exit,
                               const float* pose_jac,
                               const float* scale_jac,
                               const float* point_jac,
                               const float* pose_rhs,
                               const float* pose_diag,
                               const float* pose_tril,
                               const float* scale_rhs,
                               const float* scale_diag,
                               const float* point_rhs,
                               const float* point_diag,
                               const float* point_tril,
                               float* pose_step,
                               float* scale_step,
                               float* point_step) {
  return impl_->SolveStep(diag,
                          pcg_iteration_max,
                          pcg_relative_error_exit,
                          pose_jac,
                          scale_jac,
                          point_jac,
                          pose_rhs,
                          pose_diag,
                          pose_tril,
                          scale_rhs,
                          scale_diag,
                          point_rhs,
                          point_diag,
                          point_tril,
                          pose_step,
                          scale_step,
                          point_step);
}

}  // namespace caspar
