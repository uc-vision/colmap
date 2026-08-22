#include "colmap/exe/sfm.h"

#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/alignment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/estimators/view_graph_calibration.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/reconstruction.h"

#include <filesystem>
#include <memory>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

std::map<size_t, std::shared_ptr<Reconstruction>> ReconstructionManagerToMap(
    const std::shared_ptr<ReconstructionManager>& reconstruction_manager) {
  std::map<size_t, std::shared_ptr<Reconstruction>> reconstructions;
  for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    reconstructions[i] = reconstruction_manager->Get(i);
  }
  return reconstructions;
}

using ImageIdArray = py::array_t<image_t, py::array::c_style>;
using Point2DIndexArray = py::array_t<point2D_t, py::array::c_style>;
using Point2DArray = py::array_t<float, py::array::c_style>;

class PreparedGlobalMapping {
 public:
  PreparedGlobalMapping(Database& database, GlobalPipelineOptions options)
      : options_(std::move(options)),
        database_cache_(CreateDatabaseCache(database, options_)),
        reconstruction_(std::make_shared<Reconstruction>()),
        mapper_options_(CreateMapperOptions(options_)),
        global_mapper_(database_cache_) {
    if (options_.decompose_relative_pose) {
      const int num_threads =
          options_.mapper.refine_sensor_from_rig ? 1 : options_.num_threads;
      MaybeDecomposeRelativePoses(database_cache_.get(), num_threads);
    }
    global_mapper_.BeginReconstruction(reconstruction_);
    THROW_CHECK(
        global_mapper_.RotationAveraging(mapper_options_.RotationAveraging()));
    global_mapper_.EstablishTracks(mapper_options_);
  }

  py::dict TrackArrays() const { return ::TrackArrays(*reconstruction_); }

  std::shared_ptr<Reconstruction> Finish(
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
      const point2D_t point2D_idx = point2D_indices_view(index);
      const Eigen::Vector2d point(xy_view(index, 0), xy_view(index, 1));
      database_cache_->Image(image_id).Point2D(point2D_idx).xy = point;
      reconstruction_->Image(image_id).Point2D(point2D_idx).xy = point;
    }

    GlobalMapperOptions finish_options = mapper_options_;
    finish_options.skip_rotation_averaging = true;
    finish_options.skip_track_establishment = true;
    THROW_CHECK(global_mapper_.Solve(finish_options));
    AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                       reconstruction_.get());
    if (options_.extract_colors && !options_.image_path.empty()) {
      reconstruction_->ExtractColorsForAllImages(options_.image_path,
                                                 options_.num_threads);
    }
    return reconstruction_;
  }

 private:
  static std::shared_ptr<DatabaseCache> CreateDatabaseCache(
      Database& database, const GlobalPipelineOptions& options) {
    DatabaseCache::Options cache_options;
    cache_options.min_num_matches = options.min_num_matches;
    cache_options.ignore_watermarks = options.ignore_watermarks;
    cache_options.image_names = {options.image_names.begin(),
                                 options.image_names.end()};
    return DatabaseCache::Create(database, cache_options);
  }

  static GlobalMapperOptions CreateMapperOptions(
      const GlobalPipelineOptions& options) {
    GlobalMapperOptions mapper_options = options.mapper;
    mapper_options.image_path = options.image_path;
    mapper_options.num_threads = options.num_threads;
    mapper_options.random_seed = options.random_seed;
    return mapper_options;
  }

  const GlobalPipelineOptions options_;
  const std::shared_ptr<DatabaseCache> database_cache_;
  const std::shared_ptr<Reconstruction> reconstruction_;
  const GlobalMapperOptions mapper_options_;
  GlobalMapper global_mapper_;
};

}  // namespace

std::shared_ptr<Reconstruction> TriangulatePoints(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const bool clear_points,
    const IncrementalPipelineOptions& options,
    const bool refine_intrinsics) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           image_path,
                           output_path,
                           options,
                           clear_points,
                           refine_intrinsics);
  return reconstruction;
}

std::map<size_t, std::shared_ptr<Reconstruction>> IncrementalMapping(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const IncrementalPipelineOptions& options,
    const std::filesystem::path& input_path,
    std::function<void()> initial_image_pair_callback,
    std::function<void()> next_image_callback) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    reconstruction_manager->Read(input_path);
  }
  auto options_ = std::make_shared<IncrementalPipelineOptions>(options);

  PyInterrupt py_interrupt(1.0);  // Check for interrupts every second
  auto next_image_callback_py_interruptible =
      [&py_interrupt, next_image_callback = std::move(next_image_callback)]() {
        if (py_interrupt.Raised()) {
          throw py::error_already_set();
        }
        if (next_image_callback) {
          next_image_callback();
        }
      };

  if (!RunIncrementalMapperImpl(database_path,
                                image_path,
                                output_path,
                                options_,
                                reconstruction_manager,
                                initial_image_pair_callback,
                                next_image_callback_py_interruptible)) {
    return {};
  }

  return ReconstructionManagerToMap(reconstruction_manager);
}

std::map<size_t, std::shared_ptr<Reconstruction>> GlobalMapping(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    GlobalPipelineOptions options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options_ = std::make_shared<GlobalPipelineOptions>(std::move(options));
  if (!RunGlobalMapperImpl(database_path,
                           image_path,
                           output_path,
                           options_,
                           reconstruction_manager)) {
    return {};
  }

  return ReconstructionManagerToMap(reconstruction_manager);
}

void BundleAdjustment(const std::shared_ptr<Reconstruction>& reconstruction,
                      const BundleAdjustmentOptions& options) {
  py::gil_scoped_release release;
  OptionManager option_manager;
  option_manager.bundle_adjustment =
      std::make_shared<BundleAdjustmentOptions>(options);
  BundleAdjustmentController controller(option_manager, reconstruction);
  controller.Run();
}

bool ViewGraphCalibration(const std::filesystem::path& database_path,
                          const ViewGraphCalibrationOptions& options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  py::gil_scoped_release release;
  auto database = Database::Open(database_path);
  return CalibrateViewGraph(options, database.get());
}

void BindSfM(py::module& m) {
  // ViewGraphCalibrationOptions
  {
    using Opts = ViewGraphCalibrationOptions;
    auto PyOpts =
        py::classh<Opts>(m, "ViewGraphCalibrationOptions")
            .def(py::init<>())
            .def_readwrite("random_seed", &Opts::random_seed)
            .def_readwrite("cross_validate_prior_focal_lengths",
                           &Opts::cross_validate_prior_focal_lengths)
            .def_readwrite("min_calibrated_pair_ratio",
                           &Opts::min_calibrated_pair_ratio)
            .def_readwrite("reestimate_relative_pose",
                           &Opts::reestimate_relative_pose)
            .def_readwrite("min_focal_length_ratio",
                           &Opts::min_focal_length_ratio)
            .def_readwrite("max_focal_length_ratio",
                           &Opts::max_focal_length_ratio)
            .def_readwrite("max_calibration_error",
                           &Opts::max_calibration_error)
            .def_readwrite("loss_function_scale", &Opts::loss_function_scale)
            .def_readwrite("relpose_max_error", &Opts::relpose_max_error)
            .def_readwrite("relpose_min_num_inliers",
                           &Opts::relpose_min_num_inliers)
            .def_readwrite("relpose_min_inlier_ratio",
                           &Opts::relpose_min_inlier_ratio);
    MakeDataclass(PyOpts);
  }

  // GlobalMapperOptions
  {
    using Opts = GlobalMapperOptions;
    auto PyOpts =
        py::classh<Opts>(m, "GlobalMapperOptions")
            .def(py::init<>())
            .def_readwrite("num_threads", &Opts::num_threads)
            .def_readwrite("random_seed", &Opts::random_seed)
            .def_readwrite("refine_sensor_from_rig",
                           &Opts::refine_sensor_from_rig,
                           "When False, treat each non-ref sensor's "
                           "cam_from_rig as a pre-calibrated constant across "
                           "rotation averaging, global positioning and "
                           "bundle adjustment.")
            .def_readwrite("rotation_averaging", &Opts::rotation_averaging)
            .def_readwrite("global_positioning", &Opts::global_positioning)
            .def_readwrite("bundle_adjustment", &Opts::bundle_adjustment)
            .def_readwrite("retriangulation", &Opts::retriangulation)
            .def_readwrite("track_intra_image_consistency_threshold",
                           &Opts::track_intra_image_consistency_threshold)
            .def_readwrite("track_required_tracks_per_view",
                           &Opts::track_required_tracks_per_view)
            .def_readwrite("track_min_num_views_per_track",
                           &Opts::track_min_num_views_per_track)
            .def_readwrite("keep_max_num_tracks", &Opts::keep_max_num_tracks)
            .def_readwrite("max_angular_reproj_error_deg",
                           &Opts::max_angular_reproj_error_deg)
            .def_readwrite("max_normalized_reproj_error",
                           &Opts::max_normalized_reproj_error)
            .def_readwrite("min_tri_angle_deg", &Opts::min_tri_angle_deg)
            .def_readwrite("ba_num_iterations", &Opts::ba_num_iterations)
            .def_readwrite("retriangulation_max_num_refinements",
                           &Opts::retriangulation_max_num_refinements)
            .def_readwrite("retriangulation_ba_max_num_iterations",
                           &Opts::retriangulation_ba_max_num_iterations)
            .def_readwrite("ba_skip_fixed_rotation_stage",
                           &Opts::ba_skip_fixed_rotation_stage)
            .def_readwrite("ba_skip_joint_optimization_stage",
                           &Opts::ba_skip_joint_optimization_stage)
            .def_readwrite("skip_rotation_averaging",
                           &Opts::skip_rotation_averaging)
            .def_readwrite("skip_track_establishment",
                           &Opts::skip_track_establishment)
            .def_readwrite("skip_global_positioning",
                           &Opts::skip_global_positioning)
            .def_readwrite("skip_bundle_adjustment",
                           &Opts::skip_bundle_adjustment)
            .def_readwrite("skip_retriangulation", &Opts::skip_retriangulation);
    MakeDataclass(PyOpts);
  }

  // GlobalPipelineOptions
  {
    using Opts = GlobalPipelineOptions;
    auto PyOpts =
        py::classh<Opts>(m, "GlobalPipelineOptions")
            .def(py::init<>())
            .def_readwrite("min_num_matches", &Opts::min_num_matches)
            .def_readwrite("ignore_watermarks", &Opts::ignore_watermarks)
            .def_readwrite("image_names", &Opts::image_names)
            .def_readwrite("image_path", &Opts::image_path)
            .def_readwrite("extract_colors", &Opts::extract_colors)
            .def_readwrite("num_threads", &Opts::num_threads)
            .def_readwrite("random_seed", &Opts::random_seed)
            .def_readwrite("decompose_relative_pose",
                           &Opts::decompose_relative_pose)
            .def_readwrite("mapper", &Opts::mapper);
    MakeDataclass(PyOpts);
  }

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
        THROW_CHECK(!options.mapper.skip_rotation_averaging);
        THROW_CHECK(!options.mapper.skip_track_establishment);
        py::gil_scoped_release release;
        return std::make_unique<PreparedGlobalMapping>(database,
                                                       std::move(options));
      },
      "database"_a,
      "options"_a,
      "Establish verified feature tracks for coordinate refinement.");

  m.def("triangulate_points",
        &TriangulatePoints,
        "reconstruction"_a,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        "clear_points"_a = true,
        py::arg_v("options",
                  IncrementalPipelineOptions(),
                  "IncrementalPipelineOptions()"),
        "refine_intrinsics"_a = false,
        "Triangulate 3D points from known camera poses");

  m.def("incremental_mapping",
        &IncrementalMapping,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        py::arg_v("options",
                  IncrementalPipelineOptions(),
                  "IncrementalPipelineOptions()"),
        "input_path"_a = py::str(""),
        "initial_image_pair_callback"_a = py::none(),
        "next_image_callback"_a = py::none(),
        "Recover 3D points and unknown camera poses");

  m.def(
      "global_mapping",
      &GlobalMapping,
      "database_path"_a,
      "image_path"_a,
      "output_path"_a,
      py::arg_v("options", GlobalPipelineOptions(), "GlobalPipelineOptions()"),
      "Recover 3D points and camera poses using global SfM (GLOMAP)");

  m.def("calibrate_view_graph",
        &ViewGraphCalibration,
        "database_path"_a,
        py::arg_v("options",
                  ViewGraphCalibrationOptions(),
                  "ViewGraphCalibrationOptions()"),
        "Calibrate focal lengths from fundamental matrices and upgrade "
        "two-view geometries to CALIBRATED in the database. Run before "
        "global_mapping when reliable intrinsics are unavailable.");

  m.def("bundle_adjustment",
        &BundleAdjustment,
        "reconstruction"_a,
        py::arg_v(
            "options", BundleAdjustmentOptions(), "BundleAdjustmentOptions()"),
        "Jointly refine 3D points and camera poses");
}
