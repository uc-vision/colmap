#include "colmap/estimators/point_refinement_caspar.h"

#include <iostream>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

using Matrix3x4f = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

Matrix3x4f ProjectionMatrix(const float camera_center_x) {
  Matrix3x4f projection;
  projection << 400.0f, 0.0f, 320.0f, -400.0f * camera_center_x, 0.0f, 400.0f,
      240.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
  return projection;
}

Eigen::Vector2f Project(const Matrix3x4f& projection,
                        const Eigen::Vector3f& point) {
  const Eigen::Vector3f homogeneous = projection * point.homogeneous();
  return homogeneous.head<2>() / homogeneous.z();
}

std::vector<float> Flatten(const std::vector<Matrix3x4f>& matrices) {
  std::vector<float> values;
  values.reserve(matrices.size() * 12);
  for (const Matrix3x4f& matrix : matrices) {
    values.insert(values.end(), matrix.data(), matrix.data() + 12);
  }
  return values;
}

std::vector<float> Observations(const std::vector<Matrix3x4f>& projections,
                                const std::vector<Eigen::Vector3f>& points,
                                const std::vector<uint32_t>& point_indices,
                                const std::vector<uint32_t>& image_indices) {
  std::vector<float> pixels;
  pixels.reserve(point_indices.size() * 2);
  for (size_t index = 0; index < point_indices.size(); ++index) {
    const Eigen::Vector2f pixel = Project(projections[image_indices[index]],
                                          points[point_indices[index]]);
    pixels.insert(pixels.end(), pixel.data(), pixel.data() + 2);
  }
  return pixels;
}

TEST(CasparPointRefinement, RecoversKnownPointsFromIndexedFixedCameras) {
  const std::vector<Matrix3x4f> projections = {
      ProjectionMatrix(-1.0f),
      ProjectionMatrix(0.0f),
      ProjectionMatrix(1.0f),
  };
  const std::vector<Eigen::Vector3f> expected_points = {
      {0.2f, -0.1f, 4.0f},
      {-0.4f, 0.3f, 5.0f},
  };
  const std::vector<float> initial_points = {
      0.7f, 0.2f, 3.2f, -0.8f, 0.0f, 4.1f};
  const std::vector<uint32_t> point_indices = {0, 1, 0, 1, 0, 1};
  const std::vector<uint32_t> image_indices = {0, 0, 1, 1, 2, 2};
  const std::vector<float> projection_data = Flatten(projections);
  const std::vector<float> pixels =
      Observations(projections, expected_points, point_indices, image_indices);
  const std::vector<float> original_projection_data = projection_data;

  CasparPointRefinementOptions options;
  options.solver_iter_max = 100;
  const CasparPointRefinementResult result =
      RefineFixedCameraPinholePointsCaspar(initial_points.data(),
                                           expected_points.size(),
                                           projection_data.data(),
                                           projections.size(),
                                           point_indices.data(),
                                           image_indices.data(),
                                           pixels.data(),
                                           point_indices.size(),
                                           options);

  for (size_t index = 0; index < expected_points.size(); ++index) {
    const Eigen::Map<const Eigen::Vector3f> actual(result.points.data() +
                                                   index * 3);
    EXPECT_TRUE(actual.isApprox(expected_points[index], 1e-3f));
  }
  EXPECT_LT(result.summary->final_score, result.summary->initial_score);
  EXPECT_EQ(result.summary->num_residuals, 12);
  EXPECT_EQ(projection_data, original_projection_data);
}

TEST(CasparPointRefinement, SupportsMoreImagesThanObservations) {
  constexpr size_t kNumImages = 4096;
  std::vector<Matrix3x4f> projections(kNumImages, ProjectionMatrix(0.0f));
  projections[0] = ProjectionMatrix(-1.0f);
  projections[1] = ProjectionMatrix(1.0f);
  const std::vector<Eigen::Vector3f> expected_points = {
      {0.2f, -0.1f, 4.0f},
  };
  const std::vector<float> initial_points = {0.5f, 0.1f, 3.5f};
  const std::vector<uint32_t> point_indices = {0, 0};
  const std::vector<uint32_t> image_indices = {0, 1};
  const std::vector<float> projection_data = Flatten(projections);
  const std::vector<float> pixels =
      Observations(projections, expected_points, point_indices, image_indices);

  CasparPointRefinementOptions options;
  options.solver_iter_max = 100;
  const CasparPointRefinementResult result =
      RefineFixedCameraPinholePointsCaspar(initial_points.data(),
                                           1,
                                           projection_data.data(),
                                           kNumImages,
                                           point_indices.data(),
                                           image_indices.data(),
                                           pixels.data(),
                                           point_indices.size(),
                                           options);

  const Eigen::Map<const Eigen::Vector3f> actual(result.points.data());
  EXPECT_TRUE(actual.isApprox(expected_points[0], 1e-3f));
}

TEST(CasparPointRefinement, MediumProblemHasBoundedPackedAllocation) {
  constexpr size_t kNumPoints = 100000;
  constexpr size_t kNumImages = 64;
  constexpr size_t kObservationsPerPoint = 4;
  constexpr size_t kMaxPackedAllocation =
      48 * 1024 * 1024 * sizeof(StorageType) / sizeof(float);
  const size_t num_observations = kNumPoints * kObservationsPerPoint;

  std::vector<Matrix3x4f> projections;
  projections.reserve(kNumImages);
  for (size_t image_index = 0; image_index < kNumImages; ++image_index) {
    projections.push_back(
        ProjectionMatrix(6.0f * static_cast<float>(image_index) /
                             static_cast<float>(kNumImages - 1) -
                         3.0f));
  }
  const std::vector<float> projection_data = Flatten(projections);

  std::vector<Eigen::Vector3f> expected_points(kNumPoints);
  std::vector<float> initial_points(kNumPoints * 3);
  for (size_t point_index = 0; point_index < kNumPoints; ++point_index) {
    expected_points[point_index] =
        Eigen::Vector3f(0.002f * static_cast<float>(point_index % 101) - 0.1f,
                        0.002f * static_cast<float>(point_index % 79) - 0.08f,
                        4.0f + 0.05f * static_cast<float>(point_index % 11));
    Eigen::Map<Eigen::Vector3f>(initial_points.data() + point_index * 3) =
        expected_points[point_index] + Eigen::Vector3f(0.03f, -0.02f, 0.2f);
  }

  std::vector<uint32_t> point_indices(num_observations);
  std::vector<uint32_t> image_indices(num_observations);
  std::vector<float> pixels(num_observations * 2);
  for (size_t point_index = 0; point_index < kNumPoints; ++point_index) {
    for (size_t local_index = 0; local_index < kObservationsPerPoint;
         ++local_index) {
      const size_t observation_index =
          point_index * kObservationsPerPoint + local_index;
      const size_t image_index = (point_index + 11 * local_index) % kNumImages;
      point_indices[observation_index] = point_index;
      image_indices[observation_index] = image_index;
      Eigen::Map<Eigen::Vector2f>(pixels.data() + observation_index * 2) =
          Project(projections[image_index], expected_points[point_index]);
    }
  }

  CasparPointRefinementOptions options;
  options.solver_iter_max = 30;
  const CasparPointRefinementResult result =
      RefineFixedCameraPinholePointsCaspar(initial_points.data(),
                                           kNumPoints,
                                           projection_data.data(),
                                           kNumImages,
                                           point_indices.data(),
                                           image_indices.data(),
                                           pixels.data(),
                                           num_observations,
                                           options);

  for (size_t point_index = 0; point_index < kNumPoints; point_index += 997) {
    const Eigen::Map<const Eigen::Vector3f> actual(result.points.data() +
                                                   point_index * 3);
    EXPECT_TRUE(actual.isApprox(expected_points[point_index], 2e-3f));
  }
  EXPECT_LT(result.summary->allocation_size, kMaxPackedAllocation);
  std::cout << "CASPAR point refinement: " << kNumPoints << " points, "
            << num_observations << " observations, "
            << result.summary->allocation_size / (1024.0 * 1024.0) << " MiB, "
            << 1000.0 * result.summary->runtime << " ms\n";
}

}  // namespace
}  // namespace colmap
