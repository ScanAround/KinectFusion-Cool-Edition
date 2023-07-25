#pragma once

#include <eigen3/Eigen/Dense> // for Eigen::Vector3d
#include <limits>      // for std::numeric_limits

namespace kinect_fusion {

// Voxel class updated to store its position
class Voxel {
public:
  double tsdfValue;  // The TSDF value
  double weight;  // weight for the distance
  Eigen::Vector3d position; // The position of the voxel in the global frame

  Voxel();
};

}