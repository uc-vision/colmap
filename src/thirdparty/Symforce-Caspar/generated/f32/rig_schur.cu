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
#include <cusolverDn.h>

namespace caspar {
namespace {

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void CheckCusolver(cusolverStatus_t status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("cuSOLVER error " +
                             std::to_string(static_cast<int>(status)));
  }
}

class CusolverHandle {
 public:
  CusolverHandle() = default;
  CusolverHandle(const CusolverHandle&) = delete;
  CusolverHandle& operator=(const CusolverHandle&) = delete;

  ~CusolverHandle() {
    if (handle_ != nullptr) {
      cusolverDnDestroy(handle_);
    }
  }

  void Create() { CheckCusolver(cusolverDnCreate(&handle_)); }
  cusolverDnHandle_t get() const { return handle_; }

 private:
  cusolverDnHandle_t handle_ = nullptr;
};

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

struct PairRecord {
  size_t row_edge;
  size_t col_edge;
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

__global__ void InitializePoseSystemKernel(const unsigned int* active_poses,
                                           size_t active_pose_num,
                                           size_t pose_num,
                                           float lm_diag,
                                           const float* pose_rhs,
                                           const float* pose_diag,
                                           const float* pose_tril,
                                           float* reduced_matrix,
                                           float* reduced_rhs) {
  const size_t compact_pose =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (compact_pose >= active_pose_num) {
    return;
  }

  const unsigned int pose = active_poses[compact_pose];
  const size_t dimension = 6 * active_pose_num;
  for (unsigned int row = 0; row < 6; ++row) {
    reduced_rhs[6 * compact_pose + row] =
        LoadNodeValue(pose_rhs, pose_num, pose, row);
    for (unsigned int col = 0; col <= row; ++col) {
      float value;
      if (row == col) {
        value = fmaf(1.0f + lm_diag,
                     LoadNodeValue(pose_diag, pose_num, pose, row),
                     lm_diag * 1e-6f);
      } else {
        value = LoadPoseTril(pose_tril, pose_num, pose, PoseTrilSlot(row, col));
      }
      reduced_matrix[6 * compact_pose + row +
                     dimension * (6 * compact_pose + col)] = value;
    }
  }
}

__global__ void ReducePoseRhsKernel(const size_t* pose_edge_offsets,
                                    const size_t* pose_edges,
                                    size_t active_pose_num,
                                    const unsigned int* edge_points,
                                    const float* point_y,
                                    const float* edge_transform,
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
    reduced_rhs[6 * pose + pose_dim] -= correction;
  }
}

__global__ void ReducePosePairsKernel(const size_t* pair_offsets,
                                      const PairRecord* pair_records,
                                      const size_t* pair_row_poses,
                                      const size_t* pair_col_poses,
                                      const float* edge_transform,
                                      size_t active_pose_num,
                                      float* reduced_matrix) {
  const size_t pair = blockIdx.x;
  const unsigned int element = threadIdx.x;
  if (element >= 36) {
    return;
  }

  const size_t row_pose = pair_row_poses[pair];
  const size_t col_pose = pair_col_poses[pair];
  const unsigned int row_dim = element / 6;
  const unsigned int col_dim = element % 6;
  if (row_pose == col_pose && row_dim < col_dim) {
    return;
  }

  float value = 0.0f;
  for (size_t position = pair_offsets[pair]; position < pair_offsets[pair + 1];
       ++position) {
    const PairRecord record = pair_records[position];
    for (unsigned int point_dim = 0; point_dim < 3; ++point_dim) {
      value += edge_transform[18 * record.row_edge + 6 * point_dim + row_dim] *
               edge_transform[18 * record.col_edge + 6 * point_dim + col_dim];
    }
  }

  const size_t dimension = 6 * active_pose_num;
  const size_t row = 6 * row_pose + row_dim;
  const size_t col = 6 * col_pose + col_dim;
  reduced_matrix[row + dimension * col] -= value;
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
                                           const float* reduced_step,
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

  const float* chol = point_chol + 6 * static_cast<size_t>(point);
  const float x2 = value[2] / chol[5];
  const float x1 = (value[1] - chol[4] * x2) / chol[2];
  const float x0 = (value[0] - chol[1] * x1 - chol[3] * x2) / chol[0];
  const size_t point_offset = 4 * static_cast<size_t>(point);
  point_step[point_offset] = x0;
  point_step[point_offset + 1] = x1;
  point_step[point_offset + 2] = x2;
}

size_t PairIndex(size_t row, size_t col) { return row * (row + 1) / 2 + col; }

}  // namespace

class RigSchurSolver::Impl {
 public:
  Impl(int device_id,
       size_t pose_num,
       size_t point_num,
       const std::vector<unsigned int>& pose_indices,
       const std::vector<unsigned int>& point_indices,
       const std::vector<unsigned int>& fixed_pose_point_indices,
       const std::vector<unsigned int>& fixed_point_pose_indices)
      : device_id_(device_id),
        pose_num_(pose_num),
        point_num_(point_num),
        factor_num_(pose_indices.size()) {
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
    const size_t max_generated_count =
        std::numeric_limits<unsigned int>::max();
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
    std::vector<unsigned int> active_points;
    for (size_t point = 0; point < point_num_; ++point) {
      if (active_point_flags[point]) {
        active_points.push_back(static_cast<unsigned int>(point));
      }
    }
    active_pose_num_ = active_poses.size();
    active_point_num_ = active_points.size();
    if (active_pose_num_ >
        static_cast<size_t>(std::numeric_limits<int>::max()) / 6) {
      throw std::invalid_argument("Rig Schur reduced system is too large");
    }
    const size_t dimension = 6 * active_pose_num_;
    dimension_ = static_cast<int>(dimension);

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

    pair_block_num_ = active_pose_num_ * (active_pose_num_ + 1) / 2;
    std::vector<size_t> pair_offsets(pair_block_num_ + 1, 0);
    for (size_t point = 0; point < point_num_; ++point) {
      for (size_t row_edge = point_edge_offsets[point];
           row_edge < point_edge_offsets[point + 1];
           ++row_edge) {
        for (size_t col_edge = point_edge_offsets[point]; col_edge <= row_edge;
             ++col_edge) {
          ++pair_offsets[PairIndex(edge_poses[row_edge], edge_poses[col_edge]) +
                         1];
        }
      }
    }
    std::partial_sum(
        pair_offsets.begin(), pair_offsets.end(), pair_offsets.begin());
    std::vector<PairRecord> pair_records(pair_offsets.back());
    std::vector<size_t> pair_cursors = pair_offsets;
    for (size_t point = 0; point < point_num_; ++point) {
      for (size_t row_edge = point_edge_offsets[point];
           row_edge < point_edge_offsets[point + 1];
           ++row_edge) {
        for (size_t col_edge = point_edge_offsets[point]; col_edge <= row_edge;
             ++col_edge) {
          const size_t pair =
              PairIndex(edge_poses[row_edge], edge_poses[col_edge]);
          pair_records[pair_cursors[pair]++] = {row_edge, col_edge};
        }
      }
    }
    std::vector<size_t> pair_row_poses;
    std::vector<size_t> pair_col_poses;
    pair_row_poses.reserve(pair_block_num_);
    pair_col_poses.reserve(pair_block_num_);
    for (size_t row = 0; row < active_pose_num_; ++row) {
      for (size_t col = 0; col <= row; ++col) {
        pair_row_poses.push_back(row);
        pair_col_poses.push_back(col);
      }
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
    pair_offsets_ = DeviceBuffer<size_t>(pair_offsets);
    pair_records_ = DeviceBuffer<PairRecord>(pair_records);
    pair_row_poses_ = DeviceBuffer<size_t>(pair_row_poses);
    pair_col_poses_ = DeviceBuffer<size_t>(pair_col_poses);
    point_chol_ = DeviceBuffer<float>(6 * point_num_);
    point_y_ = DeviceBuffer<float>(3 * point_num_);
    edge_transform_ = DeviceBuffer<float>(18 * edge_num_);
    reduced_matrix_ = DeviceBuffer<float>(dimension * dimension);
    reduced_rhs_ = DeviceBuffer<float>(dimension);
    info_ = DeviceBuffer<int>(1);

    cusolver_handle_.Create();
    int workspace_size = 0;
    CheckCusolver(cusolverDnSpotrf_bufferSize(cusolver_handle_.get(),
                                              CUBLAS_FILL_MODE_LOWER,
                                              dimension_,
                                              reduced_matrix_.data(),
                                              dimension_,
                                              &workspace_size));
    workspace_ = DeviceBuffer<float>(workspace_size);
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

  bool SolveStep(float diag,
                 const float* pose_jac,
                 const float* point_jac,
                 const float* pose_rhs,
                 const float* pose_diag,
                 const float* pose_tril,
                 const float* point_rhs,
                 const float* point_diag,
                 const float* point_tril,
                 float* pose_step,
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

    CheckCuda(cudaMemset(
        reduced_matrix_.data(), 0, reduced_matrix_.size() * sizeof(float)));
    InitializePoseSystemKernel<<<GridSize(active_pose_num_, block_size),
                                 block_size>>>(active_poses_.data(),
                                               active_pose_num_,
                                               pose_num_,
                                               diag,
                                               pose_rhs,
                                               pose_diag,
                                               pose_tril,
                                               reduced_matrix_.data(),
                                               reduced_rhs_.data());
    CheckCuda(cudaGetLastError());
    ReducePoseRhsKernel<<<GridSize(active_pose_num_, 1), 6 * 32>>>(
        pose_edge_offsets_.data(),
        pose_edges_.data(),
        active_pose_num_,
        edge_points_.data(),
        point_y_.data(),
        edge_transform_.data(),
        reduced_rhs_.data());
    CheckCuda(cudaGetLastError());
    ReducePosePairsKernel<<<GridSize(pair_block_num_, 1), 36>>>(
        pair_offsets_.data(),
        pair_records_.data(),
        pair_row_poses_.data(),
        pair_col_poses_.data(),
        edge_transform_.data(),
        active_pose_num_,
        reduced_matrix_.data());
    CheckCuda(cudaGetLastError());

    CheckCuda(cudaMemset(info_.data(), 0, sizeof(int)));
    CheckCusolver(cusolverDnSpotrf(cusolver_handle_.get(),
                                   CUBLAS_FILL_MODE_LOWER,
                                   dimension_,
                                   reduced_matrix_.data(),
                                   dimension_,
                                   workspace_.data(),
                                   static_cast<int>(workspace_.size()),
                                   info_.data()));
    CheckCuda(
        cudaMemcpy(&info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
      return false;
    }
    CheckCusolver(cusolverDnSpotrs(cusolver_handle_.get(),
                                   CUBLAS_FILL_MODE_LOWER,
                                   dimension_,
                                   1,
                                   reduced_matrix_.data(),
                                   dimension_,
                                   reduced_rhs_.data(),
                                   dimension_,
                                   info_.data()));
    CheckCuda(
        cudaMemcpy(&info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
      throw std::runtime_error("cuSOLVER potrs failed");
    }

    ScatterPoseStepKernel<<<GridSize(active_pose_num_, block_size),
                            block_size>>>(active_poses_.data(),
                                          active_pose_num_,
                                          pose_num_,
                                          reduced_rhs_.data(),
                                          pose_step);
    CheckCuda(cudaGetLastError());
    BackSubstitutePointsKernel<<<GridSize(active_point_num_, block_size),
                                 block_size>>>(active_points_.data(),
                                               active_point_num_,
                                               point_num_,
                                               point_edge_offsets_.data(),
                                               edge_poses_.data(),
                                               point_chol_.data(),
                                               point_y_.data(),
                                               edge_transform_.data(),
                                               reduced_rhs_.data(),
                                               point_step);
    CheckCuda(cudaGetLastError());
    return true;
  }

 private:
  int device_id_;
  size_t pose_num_;
  size_t point_num_;
  size_t factor_num_;
  size_t active_pose_num_;
  size_t active_point_num_;
  size_t edge_num_;
  size_t pair_block_num_;
  size_t max_grid_size_;
  int dimension_;
  DeviceBuffer<unsigned int> active_poses_;
  DeviceBuffer<unsigned int> active_points_;
  DeviceBuffer<unsigned int> edge_points_;
  DeviceBuffer<size_t> edge_poses_;
  DeviceBuffer<size_t> edge_factor_offsets_;
  DeviceBuffer<size_t> edge_factor_indices_;
  DeviceBuffer<size_t> point_edge_offsets_;
  DeviceBuffer<size_t> pose_edge_offsets_;
  DeviceBuffer<size_t> pose_edges_;
  DeviceBuffer<size_t> pair_offsets_;
  DeviceBuffer<PairRecord> pair_records_;
  DeviceBuffer<size_t> pair_row_poses_;
  DeviceBuffer<size_t> pair_col_poses_;
  DeviceBuffer<float> point_chol_;
  DeviceBuffer<float> point_y_;
  DeviceBuffer<float> edge_transform_;
  DeviceBuffer<float> reduced_matrix_;
  DeviceBuffer<float> reduced_rhs_;
  DeviceBuffer<float> workspace_;
  DeviceBuffer<int> info_;
  CusolverHandle cusolver_handle_;
};

RigSchurSolver::RigSchurSolver(
    int device_id,
    size_t pose_num,
    size_t point_num,
    const std::vector<unsigned int>& pose_indices,
    const std::vector<unsigned int>& point_indices,
    const std::vector<unsigned int>& fixed_pose_point_indices,
    const std::vector<unsigned int>& fixed_point_pose_indices)
    : impl_(std::make_unique<Impl>(device_id,
                                   pose_num,
                                   point_num,
                                   pose_indices,
                                   point_indices,
                                   fixed_pose_point_indices,
                                   fixed_point_pose_indices)) {}

RigSchurSolver::~RigSchurSolver() = default;

bool RigSchurSolver::SolveStep(float diag,
                               const float* pose_jac,
                               const float* point_jac,
                               const float* pose_rhs,
                               const float* pose_diag,
                               const float* pose_tril,
                               const float* point_rhs,
                               const float* point_diag,
                               const float* point_tril,
                               float* pose_step,
                               float* point_step) {
  return impl_->SolveStep(diag,
                          pose_jac,
                          point_jac,
                          pose_rhs,
                          pose_diag,
                          pose_tril,
                          point_rhs,
                          point_diag,
                          point_tril,
                          pose_step,
                          point_step);
}

}  // namespace caspar
