#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace caspar {
class GraphSolver;
}

namespace colmap {

struct CasparReprojectionErrorSummary {
  size_t point_count = 0;
  size_t observation_count = 0;
  double mean_pixels = 0.0;
  double median_pixels = 0.0;
  double p95_pixels = 0.0;
};

class CasparRowReprojectionValidator {
 public:
  CasparRowReprojectionValidator(size_t num_points,
                                 const float* image_from_world,
                                 size_t num_images,
                                 size_t maximum_chunk_points,
                                 size_t maximum_chunk_observations,
                                 size_t device_id);
  ~CasparRowReprojectionValidator();

  CasparRowReprojectionValidator(const CasparRowReprojectionValidator&) =
      delete;
  CasparRowReprojectionValidator& operator=(
      const CasparRowReprojectionValidator&) = delete;

  size_t DeviceId() const;

  void MeasureChunk(caspar::GraphSolver& solver,
                    size_t row_point_start,
                    size_t num_points,
                    const uint32_t* observation_offsets,
                    const uint32_t* observation_image_indices,
                    const float* observation_xy,
                    size_t num_observations);

  CasparReprojectionErrorSummary Summarize(size_t num_observations);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap
