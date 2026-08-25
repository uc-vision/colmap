#pragma once

#include "colmap/controllers/global_pipeline.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/global_mapper.h"

#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

class PreparedGlobalMapping {
 public:
  using ImageIdArray =
      pybind11::array_t<colmap::image_t, pybind11::array::c_style>;
  using Point2DIndexArray =
      pybind11::array_t<colmap::point2D_t, pybind11::array::c_style>;
  using Point2DArray = pybind11::array_t<float, pybind11::array::c_style>;

  PreparedGlobalMapping(colmap::Database& database,
                        colmap::GlobalPipelineOptions options);
  PreparedGlobalMapping(
      colmap::Database& database,
      colmap::GlobalPipelineOptions options,
      std::shared_ptr<const colmap::GlobalMapperStrategy> strategy);

  pybind11::dict TrackArrays() const;
  std::shared_ptr<colmap::Reconstruction> Finish(
      const ImageIdArray& image_ids,
      const Point2DIndexArray& point2D_indices,
      const Point2DArray& xy);

 private:
  static std::shared_ptr<colmap::DatabaseCache> CreateDatabaseCache(
      colmap::Database& database, const colmap::GlobalPipelineOptions& options);
  static colmap::GlobalMapperOptions CreateMapperOptions(
      const colmap::GlobalPipelineOptions& options);

  const colmap::GlobalPipelineOptions options_;
  const std::shared_ptr<colmap::DatabaseCache> database_cache_;
  const std::shared_ptr<colmap::Reconstruction> reconstruction_;
  const colmap::GlobalMapperOptions mapper_options_;
  colmap::GlobalMapper global_mapper_;
};

void BindPreparedGlobalMapping(pybind11::module& m);
