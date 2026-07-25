// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/feature/fixed_dimension_cuda.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace colmap {
namespace {

constexpr int kMatchBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kColumnBlockWidth = 32;
constexpr int kColumnBlockHeight = 8;
constexpr unsigned int kFullWarpMask = 0xffffffff;
static_assert(sizeof(Eigen::half) == sizeof(__half));

void CublasSafeCall(const cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    LOG(FATAL_THROW) << "cuBLAS error: " << static_cast<int>(status);
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      CUDA_SAFE_CALL(cudaFree(data_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void Reserve(const size_t size) {
    if (size <= capacity_) {
      return;
    }
    if (data_ != nullptr) {
      CUDA_SAFE_CALL(cudaFree(data_));
    }
    CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
    capacity_ = size;
  }

  T* Data() { return data_; }
  const T* Data() const { return data_; }

 private:
  T* data_ = nullptr;
  size_t capacity_ = 0;
};

struct DeviceFeatures {
  DeviceBuffer<__half> descriptors;
  DeviceBuffer<float> locations;
  int num_features = 0;
  int descriptor_dimension = 0;
  int num_locations = 0;
};

struct Matrix3x3 {
  float values[9];
};

struct Candidate {
  float score;
  int index;
};

struct TopTwo {
  Candidate best;
  Candidate second;
};

__device__ TopTwo EmptyTopTwo() {
  return TopTwo{Candidate{-CUDART_INF_F, -1}, Candidate{-CUDART_INF_F, -1}};
}

__device__ bool IsBetter(const Candidate candidate, const Candidate reference) {
  return candidate.index >= 0 &&
         (reference.index < 0 || candidate.score > reference.score ||
          (candidate.score == reference.score &&
           candidate.index < reference.index));
}

__device__ void InsertCandidate(const Candidate candidate, TopTwo* top_two) {
  if (candidate.index < 0) {
    return;
  }
  if (IsBetter(candidate, top_two->best)) {
    top_two->second = top_two->best;
    top_two->best = candidate;
  } else if (candidate.index != top_two->best.index &&
             IsBetter(candidate, top_two->second)) {
    top_two->second = candidate;
  }
}

__device__ TopTwo MergeTopTwo(TopTwo left, const TopTwo right) {
  InsertCandidate(right.best, &left);
  InsertCandidate(right.second, &left);
  return left;
}

__device__ TopTwo WarpReduceTopTwo(TopTwo top_two) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const TopTwo other{
        Candidate{__shfl_down_sync(kFullWarpMask, top_two.best.score, offset),
                  __shfl_down_sync(kFullWarpMask, top_two.best.index, offset)},
        Candidate{
            __shfl_down_sync(kFullWarpMask, top_two.second.score, offset),
            __shfl_down_sync(kFullWarpMask, top_two.second.index, offset)}};
    top_two = MergeTopTwo(top_two, other);
  }
  return top_two;
}

__device__ int AcceptedMatch(const TopTwo top_two,
                             const float max_ratio,
                             const float max_distance) {
  if (top_two.best.index < 0) {
    return -1;
  }

  const float best_distance =
      acosf(fminf(fmaxf(top_two.best.score, -1.0f), 1.0f));
  if (best_distance >= max_distance) {
    return -1;
  }

  const float second_distance =
      acosf(fminf(fmaxf(top_two.second.score, -1.0f), 1.0f));
  if (best_distance >= max_ratio * second_distance) {
    return -1;
  }
  return top_two.best.index;
}

__global__ void MaskGuidedScores(const float* locations1,
                                 const float* locations2,
                                 const int num_features1,
                                 const int num_features2,
                                 const FixedDimensionGuidanceType guidance_type,
                                 const Matrix3x3 guidance,
                                 const float max_residual,
                                 float* scores) {
  const int feature2 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int feature1 = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (feature1 >= num_features1 || feature2 >= num_features2) {
    return;
  }

  const float x1 = locations1[2 * feature1];
  const float y1 = locations1[2 * feature1 + 1];
  const float x2 = locations2[2 * feature2];
  const float y2 = locations2[2 * feature2 + 1];

  bool reject;
  if (guidance_type == FixedDimensionGuidanceType::EPIPOLAR) {
    const float line2_x =
        guidance.values[0] * x1 + guidance.values[1] * y1 + guidance.values[2];
    const float line2_y =
        guidance.values[3] * x1 + guidance.values[4] * y1 + guidance.values[5];
    const float line2_z =
        guidance.values[6] * x1 + guidance.values[7] * y1 + guidance.values[8];
    const float line1_x =
        guidance.values[0] * x2 + guidance.values[3] * y2 + guidance.values[6];
    const float line1_y =
        guidance.values[1] * x2 + guidance.values[4] * y2 + guidance.values[7];
    const float numerator = x2 * line2_x + y2 * line2_y + line2_z;
    const float denominator = line2_x * line2_x + line2_y * line2_y +
                              line1_x * line1_x + line1_y * line1_y;
    reject = numerator * numerator > max_residual * denominator;
  } else {
    const float projected_z =
        guidance.values[6] * x1 + guidance.values[7] * y1 + guidance.values[8];
    const float projected_x = (guidance.values[0] * x1 +
                               guidance.values[1] * y1 + guidance.values[2]) /
                              projected_z;
    const float projected_y = (guidance.values[3] * x1 +
                               guidance.values[4] * y1 + guidance.values[5]) /
                              projected_z;
    const float residual_x = projected_x - x2;
    const float residual_y = projected_y - y2;
    reject = residual_x * residual_x + residual_y * residual_y > max_residual;
  }

  if (reject) {
    scores[static_cast<size_t>(feature1) * num_features2 + feature2] =
        -CUDART_INF_F;
  }
}

__global__ void FindRowMatches(const float* scores,
                               const int num_features1,
                               const int num_features2,
                               const float max_ratio,
                               const float max_distance,
                               int* best_matches) {
  const int feature1 = blockIdx.x;
  if (feature1 >= num_features1) {
    return;
  }

  TopTwo local = EmptyTopTwo();
  for (int feature2 = threadIdx.x; feature2 < num_features2;
       feature2 += blockDim.x) {
    InsertCandidate(
        Candidate{
            scores[static_cast<size_t>(feature1) * num_features2 + feature2],
            feature2},
        &local);
  }

  local = WarpReduceTopTwo(local);

  __shared__ TopTwo warp_top_two[kMatchBlockSize / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_top_two[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    TopTwo block_top_two =
        lane < kMatchBlockSize / kWarpSize ? warp_top_two[lane] : EmptyTopTwo();
    block_top_two = WarpReduceTopTwo(block_top_two);
    if (lane == 0) {
      best_matches[feature1] =
          AcceptedMatch(block_top_two, max_ratio, max_distance);
    }
  }
}

__global__ void FindColumnMatches(const float* scores,
                                  const int num_features1,
                                  const int num_features2,
                                  const float max_ratio,
                                  const float max_distance,
                                  int* best_matches) {
  const int feature2 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;

  TopTwo local = EmptyTopTwo();
  if (feature2 < num_features2) {
    for (int feature1 = threadIdx.y; feature1 < num_features1;
         feature1 += blockDim.y) {
      InsertCandidate(
          Candidate{
              scores[static_cast<size_t>(feature1) * num_features2 + feature2],
              feature1},
          &local);
    }
  }

  __shared__ TopTwo shared[kColumnBlockHeight][kColumnBlockWidth];
  shared[threadIdx.y][threadIdx.x] = local;
  __syncthreads();

  for (int stride = kColumnBlockHeight / 2; stride > 0; stride /= 2) {
    if (threadIdx.y < stride) {
      shared[threadIdx.y][threadIdx.x] =
          MergeTopTwo(shared[threadIdx.y][threadIdx.x],
                      shared[threadIdx.y + stride][threadIdx.x]);
    }
    __syncthreads();
  }

  if (threadIdx.y == 0 && feature2 < num_features2) {
    best_matches[feature2] =
        AcceptedMatch(shared[0][threadIdx.x], max_ratio, max_distance);
  }
}

Matrix3x3 ToMatrix3x3(const Eigen::Matrix3f& matrix) {
  Matrix3x3 result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result.values[3 * row + column] = matrix(row, column);
    }
  }
  return result;
}

}  // namespace

struct FixedDimensionCudaMatcher::Impl {
  explicit Impl(const int max_num_features)
      : max_num_features(max_num_features) {
    THROW_CHECK_GT(max_num_features, 0);
    CublasSafeCall(cublasCreate(&cublas_handle));
  }

  ~Impl() { CublasSafeCall(cublasDestroy(cublas_handle)); }

  void ComputeScores() {
    const DeviceFeatures& features1 = features[0];
    const DeviceFeatures& features2 = features[1];
    const size_t num_scores =
        static_cast<size_t>(features1.num_features) * features2.num_features;
    scores.Reserve(num_scores);

    constexpr float kAlpha = 1.0f;
    constexpr float kBeta = 0.0f;
    CublasSafeCall(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                features2.num_features,
                                features1.num_features,
                                features1.descriptor_dimension,
                                &kAlpha,
                                features2.descriptors.Data(),
                                CUDA_R_16F,
                                features2.descriptor_dimension,
                                features1.descriptors.Data(),
                                CUDA_R_16F,
                                features1.descriptor_dimension,
                                &kBeta,
                                scores.Data(),
                                CUDA_R_32F,
                                features2.num_features,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  void FindMatches(const double max_ratio,
                   const double max_distance,
                   const bool cross_check,
                   FeatureMatches* matches) {
    const int num_features1 = features[0].num_features;
    const int num_features2 = features[1].num_features;
    row_matches_device.Reserve(num_features1);
    row_matches.resize(num_features1);

    FindRowMatches<<<num_features1, kMatchBlockSize>>>(
        scores.Data(),
        num_features1,
        num_features2,
        static_cast<float>(max_ratio),
        static_cast<float>(max_distance),
        row_matches_device.Data());
    CUDA_CHECK();

    if (cross_check) {
      column_matches_device.Reserve(num_features2);
      column_matches.resize(num_features2);
      const dim3 block(kColumnBlockWidth, kColumnBlockHeight);
      const dim3 grid((num_features2 + kColumnBlockWidth - 1) /
                      kColumnBlockWidth);
      FindColumnMatches<<<grid, block>>>(scores.Data(),
                                         num_features1,
                                         num_features2,
                                         static_cast<float>(max_ratio),
                                         static_cast<float>(max_distance),
                                         column_matches_device.Data());
      CUDA_CHECK();
    }

    CUDA_SAFE_CALL(cudaMemcpy(row_matches.data(),
                              row_matches_device.Data(),
                              row_matches.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    if (cross_check) {
      CUDA_SAFE_CALL(cudaMemcpy(column_matches.data(),
                                column_matches_device.Data(),
                                column_matches.size() * sizeof(int),
                                cudaMemcpyDeviceToHost));
    }

    matches->clear();
    matches->reserve(num_features1);
    for (int feature1 = 0; feature1 < num_features1; ++feature1) {
      const int feature2 = row_matches[feature1];
      if (feature2 >= 0 &&
          (!cross_check || column_matches[feature2] == feature1)) {
        matches->emplace_back(static_cast<point2D_t>(feature1),
                              static_cast<point2D_t>(feature2));
      }
    }
  }

  const int max_num_features;
  cublasHandle_t cublas_handle = nullptr;
  std::array<DeviceFeatures, 2> features;
  DeviceBuffer<float> scores;
  DeviceBuffer<int> row_matches_device;
  DeviceBuffer<int> column_matches_device;
  std::vector<int> row_matches;
  std::vector<int> column_matches;
};

FixedDimensionCudaMatcher::FixedDimensionCudaMatcher(const int max_num_features)
    : impl_(std::make_unique<Impl>(max_num_features)) {}

FixedDimensionCudaMatcher::~FixedDimensionCudaMatcher() = default;

void FixedDimensionCudaMatcher::SetDescriptors(
    const int image_index, const FeatureDescriptorsData& descriptors) {
  THROW_CHECK_GE(image_index, 0);
  THROW_CHECK_LT(image_index, impl_->features.size());
  THROW_CHECK_EQ(descriptors.cols() % sizeof(__half), 0);

  DeviceFeatures& features = impl_->features[image_index];
  features.descriptor_dimension = descriptors.cols() / sizeof(__half);
  features.num_features =
      features.descriptor_dimension == 0
          ? 0
          : static_cast<int>(std::min<Eigen::Index>(descriptors.rows(),
                                                    impl_->max_num_features));
  const size_t size = static_cast<size_t>(features.num_features) *
                      features.descriptor_dimension;
  if (size == 0) {
    return;
  }

  features.descriptors.Reserve(size);
  CUDA_SAFE_CALL(cudaMemcpy(features.descriptors.Data(),
                            descriptors.data(),
                            size * sizeof(__half),
                            cudaMemcpyHostToDevice));
}

void FixedDimensionCudaMatcher::SetFeatureLocations(
    const int image_index, const FixedDimensionFeatureLocations& locations) {
  THROW_CHECK_GE(image_index, 0);
  THROW_CHECK_LT(image_index, impl_->features.size());

  DeviceFeatures& features = impl_->features[image_index];
  features.num_locations = static_cast<int>(
      std::min<Eigen::Index>(locations.rows(), impl_->max_num_features));
  const size_t size = static_cast<size_t>(features.num_locations) * 2;
  if (size == 0) {
    return;
  }

  features.locations.Reserve(size);
  CUDA_SAFE_CALL(cudaMemcpy(features.locations.Data(),
                            locations.data(),
                            size * sizeof(float),
                            cudaMemcpyHostToDevice));
}

void FixedDimensionCudaMatcher::Match(const double max_ratio,
                                      const double max_distance,
                                      const bool cross_check,
                                      FeatureMatches* matches) {
  THROW_CHECK_NOTNULL(matches);
  matches->clear();
  if (impl_->features[0].num_features == 0 ||
      impl_->features[1].num_features == 0) {
    return;
  }
  THROW_CHECK_EQ(impl_->features[0].descriptor_dimension,
                 impl_->features[1].descriptor_dimension);

  impl_->ComputeScores();
  impl_->FindMatches(max_ratio, max_distance, cross_check, matches);
}

void FixedDimensionCudaMatcher::MatchGuided(
    const double max_ratio,
    const double max_distance,
    const bool cross_check,
    const FixedDimensionGuidanceType guidance_type,
    const Eigen::Matrix3f& guidance,
    const float max_residual,
    FeatureMatches* matches) {
  THROW_CHECK_NOTNULL(matches);
  matches->clear();
  if (impl_->features[0].num_features == 0 ||
      impl_->features[1].num_features == 0) {
    return;
  }
  THROW_CHECK_EQ(impl_->features[0].descriptor_dimension,
                 impl_->features[1].descriptor_dimension);
  THROW_CHECK_EQ(impl_->features[0].num_locations,
                 impl_->features[0].num_features);
  THROW_CHECK_EQ(impl_->features[1].num_locations,
                 impl_->features[1].num_features);

  impl_->ComputeScores();
  const dim3 block(kColumnBlockWidth, kColumnBlockHeight);
  const dim3 grid((impl_->features[1].num_features + kColumnBlockWidth - 1) /
                      kColumnBlockWidth,
                  (impl_->features[0].num_features + kColumnBlockHeight - 1) /
                      kColumnBlockHeight);
  MaskGuidedScores<<<grid, block>>>(impl_->features[0].locations.Data(),
                                    impl_->features[1].locations.Data(),
                                    impl_->features[0].num_features,
                                    impl_->features[1].num_features,
                                    guidance_type,
                                    ToMatrix3x3(guidance),
                                    max_residual,
                                    impl_->scores.Data());
  CUDA_CHECK();

  impl_->FindMatches(max_ratio, max_distance, cross_check, matches);
}

}  // namespace colmap
