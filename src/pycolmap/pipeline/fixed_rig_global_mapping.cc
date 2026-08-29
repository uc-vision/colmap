#include "pycolmap/pipeline/fixed_rig_global_mapping.h"

#include "colmap/controllers/fixed_rig_global_pipeline.h"
#include "colmap/estimators/fixed_rig_global_positioning.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/fixed_rig_global_mapper.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pipeline/prepared_global_mapping.h"
#include "pycolmap/pybind11_extension.h"

#include <filesystem>
#include <map>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

std::map<size_t, std::shared_ptr<Reconstruction>> FixedRigGlobalMapping(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    GlobalPipelineOptions options,
    const FixedRigGlobalPositionerOptions& positioning_options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  if (!image_path.empty()) {
    THROW_CHECK_DIR_EXISTS(image_path);
  }
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  options.image_path = image_path;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  FixedRigGlobalPipeline pipeline(std::move(options),
                                  Database::Open(database_path),
                                  reconstruction_manager,
                                  positioning_options);
  pipeline.Run();
  if (reconstruction_manager->Size() == 0) {
    return {};
  }
  reconstruction_manager->Write(output_path);

  std::map<size_t, std::shared_ptr<Reconstruction>> reconstructions;
  for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    reconstructions[i] = reconstruction_manager->Get(i);
  }
  return reconstructions;
}

std::unique_ptr<PreparedGlobalMapping> PrepareFixedRigGlobalMapping(
    Database& database,
    GlobalPipelineOptions options,
    FixedRigGlobalPositionerOptions positioning_options) {
  THROW_CHECK(!options.multiple_models)
      << "Prepared global mapping produces one reconstruction";
  THROW_CHECK(!options.mapper.skip_rotation_averaging);
  THROW_CHECK(!options.mapper.skip_track_establishment);
  py::gil_scoped_release release;
  return std::make_unique<PreparedGlobalMapping>(
      database,
      std::move(options),
      CreateFixedRigGlobalMapperStrategy(std::move(positioning_options)));
}

}  // namespace

void BindFixedRigGlobalMapping(py::module& m) {
  auto PyPositioningOptions =
      py::classh<FixedRigGlobalPositionerOptions>(
          m, "FixedRigGlobalPositionerOptions")
          .def(py::init<>())
          .def_readwrite(
              "initialize_from_pose_priors",
              &FixedRigGlobalPositionerOptions::initialize_from_pose_priors,
              "Whether to seed frame positions from camera pose priors.")
          .def_readwrite(
              "require_frame_constraints",
              &FixedRigGlobalPositionerOptions::require_frame_constraints,
              "Whether incomplete frame constraints must fail rather than "
              "fall back to observation-level positioning.");
  MakeDataclass(PyPositioningOptions);

  m.def(
      "fixed_rig_global_mapping",
      &FixedRigGlobalMapping,
      "database_path"_a,
      "image_path"_a,
      "output_path"_a,
      py::arg_v("options", GlobalPipelineOptions(), "GlobalPipelineOptions()"),
      py::arg_v("positioning_options",
                FixedRigGlobalPositionerOptions(),
                "FixedRigGlobalPositionerOptions()"),
      "Recover 3D points and frame poses for a calibrated camera rig");

  m.def("prepare_fixed_rig_global_mapping",
        &PrepareFixedRigGlobalMapping,
        "database"_a,
        "options"_a,
        py::arg_v("positioning_options",
                  FixedRigGlobalPositionerOptions(),
                  "FixedRigGlobalPositionerOptions()"),
        "Establish fixed-rig feature tracks for coordinate refinement.");

  m.def(
      "run_fixed_rig_global_positioning",
      [](const GlobalPositionerOptions& options,
         const FixedRigGlobalPositionerOptions& rig_options,
         const PoseGraph& pose_graph,
         Reconstruction& reconstruction,
         const std::vector<PosePrior>& pose_priors,
         const double min_tri_angle_deg) {
        py::gil_scoped_release release;
        return RunFixedRigGlobalPositioning(options,
                                            rig_options,
                                            pose_graph,
                                            reconstruction,
                                            pose_priors,
                                            min_tri_angle_deg);
      },
      "options"_a,
      "rig_options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "pose_priors"_a,
      "min_tri_angle_deg"_a,
      "Initialize metric frame positions for a calibrated camera rig");
}
