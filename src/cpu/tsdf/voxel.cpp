#include "voxel.h"

namespace kinect_fusion {
Voxel::Voxel() : tsdfValue(std::numeric_limits<double>::quiet_NaN()), 
                 weight(std::numeric_limits<double>::quiet_NaN()), 
                 position(Eigen::Vector3d::Zero()) {}
} 