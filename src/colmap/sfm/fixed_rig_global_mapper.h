#pragma once

#include "colmap/estimators/fixed_rig_global_positioning.h"
#include "colmap/sfm/global_mapper.h"

namespace colmap {

std::shared_ptr<const GlobalMapperStrategy> CreateFixedRigGlobalMapperStrategy(
    FixedRigGlobalPositionerOptions positioner_options = {});

// Explicit calibrated-rig reconstruction facade. It reuses GlobalMapper's
// shared track, bundle-adjustment, filtering, and retriangulation stages while
// selecting fixed-rig rotation filtering and metric global positioning.
class FixedRigGlobalMapper {
 public:
  explicit FixedRigGlobalMapper(
      std::shared_ptr<const DatabaseCache> database_cache,
      FixedRigGlobalPositionerOptions positioner_options = {});

  void BeginReconstruction(
      const std::shared_ptr<Reconstruction>& reconstruction);
  bool Solve(const GlobalMapperOptions& options,
             const std::function<bool()>& on_progress = {});
  bool RotationAveraging(const GlobalMapperOptions& options);
  void EstablishTracks(const GlobalMapperOptions& options);
  std::shared_ptr<class Reconstruction> Reconstruction() const;

 private:
  GlobalMapper mapper_;
};

}  // namespace colmap
