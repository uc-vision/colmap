#include "pycolmap/pipeline/prepared_global_mapping.h"

#include "colmap/controllers/global_pipeline.h"
#include "colmap/estimators/alignment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/logging.h"

#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/reconstruction.h"

#include <utility>

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

PreparedGlobalMapping::PreparedGlobalMapping(Database& database,
                                             GlobalPipelineOptions options)
    : PreparedGlobalMapping(
          database, std::move(options), CreateGlobalMapperStrategy()) {}

PreparedGlobalMapping::PreparedGlobalMapping(
    Database& database,
    GlobalPipelineOptions options,
    std::shared_ptr<const GlobalMapperStrategy> strategy)
    : options_(std::move(options)),
      database_cache_(CreateDatabaseCache(database, options_)),
      reconstruction_(std::make_shared<Reconstruction>()),
      mapper_options_(strategy->Configure(CreateMapperOptions(options_))),
      global_mapper_(database_cache_, std::move(strategy)) {
  if (options_.decompose_relative_pose) {
    MaybeDecomposeRelativePoses(database_cache_.get());
  }
  global_mapper_.BeginReconstruction(reconstruction_);
  THROW_CHECK(
      global_mapper_.RotationAveraging(mapper_options_.RotationAveraging()));
  global_mapper_.EstablishTracks(mapper_options_);
}

py::dict PreparedGlobalMapping::TrackArrays() const {
  return ::TrackArrays(*reconstruction_);
}

std::shared_ptr<Reconstruction> PreparedGlobalMapping::Finish(
    const ImageIdArray& image_ids,
    const Point2DIndexArray& point2D_indices,
    const Point2DArray& xy) {
  THROW_CHECK_EQ(image_ids.ndim(), 1);
  THROW_CHECK_EQ(point2D_indices.ndim(), 1);
  THROW_CHECK_EQ(xy.ndim(), 2);
  THROW_CHECK_EQ(image_ids.shape(0), point2D_indices.shape(0));
  THROW_CHECK_EQ(image_ids.shape(0), xy.shape(0));
  THROW_CHECK_EQ(xy.shape(1), 2);

  const auto image_ids_view = image_ids.unchecked<1>();
  const auto point2D_indices_view = point2D_indices.unchecked<1>();
  const auto xy_view = xy.unchecked<2>();
  py::gil_scoped_release release;
  for (ssize_t index = 0; index < image_ids.shape(0); ++index) {
    const image_t image_id = image_ids_view(index);
    const point2D_t point2D_index = point2D_indices_view(index);
    const Eigen::Vector2d point(xy_view(index, 0), xy_view(index, 1));
    database_cache_->Image(image_id).Point2D(point2D_index).xy = point;
    reconstruction_->Image(image_id).Point2D(point2D_index).xy = point;
  }

  GlobalMapperOptions finish_options = mapper_options_;
  finish_options.skip_rotation_averaging = true;
  finish_options.skip_track_establishment = true;
  THROW_CHECK(global_mapper_.Solve(finish_options));
  AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                     reconstruction_.get());
  if (!options_.image_path.empty()) {
    reconstruction_->ExtractColorsForAllImages(options_.image_path,
                                               options_.num_threads);
  }
  return reconstruction_;
}

std::shared_ptr<DatabaseCache> PreparedGlobalMapping::CreateDatabaseCache(
    Database& database, const GlobalPipelineOptions& options) {
  DatabaseCache::Options cache_options;
  cache_options.min_num_matches = options.min_num_matches;
  cache_options.ignore_watermarks = options.ignore_watermarks;
  cache_options.image_names = {options.image_names.begin(),
                               options.image_names.end()};
  return DatabaseCache::Create(database, cache_options);
}

GlobalMapperOptions PreparedGlobalMapping::CreateMapperOptions(
    const GlobalPipelineOptions& options) {
  GlobalMapperOptions mapper_options = options.mapper;
  mapper_options.image_path = options.image_path;
  mapper_options.num_threads = options.num_threads;
  mapper_options.random_seed = options.random_seed;
  return mapper_options;
}

void BindPreparedGlobalMapping(py::module& m) {
  py::classh<PreparedGlobalMapping>(m, "PreparedGlobalMapping")
      .def("track_arrays", &PreparedGlobalMapping::TrackArrays)
      .def("finish",
           &PreparedGlobalMapping::Finish,
           "image_ids"_a.noconvert(),
           "point2D_indices"_a.noconvert(),
           "xy"_a.noconvert(),
           "Update track observations and complete global mapping.");

  m.def(
      "prepare_global_mapping",
      [](Database& database, GlobalPipelineOptions options) {
        THROW_CHECK(!options.multiple_models)
            << "Prepared global mapping produces one reconstruction";
        THROW_CHECK(!options.mapper.skip_rotation_averaging);
        THROW_CHECK(!options.mapper.skip_track_establishment);
        py::gil_scoped_release release;
        return std::make_unique<PreparedGlobalMapping>(database,
                                                       std::move(options));
      },
      "database"_a,
      "options"_a,
      "Establish verified feature tracks for coordinate refinement.");
}
