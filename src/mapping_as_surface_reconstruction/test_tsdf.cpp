#include "voxel_grid.h"
#include "kinect_fusion_utility.h"

int main() {

  double f_x = 517.3;  // focal length x
  double f_y = 516.5;  // focal length y
  double c_x = 318.6;  // optical center x
  double c_y = 255.3;  // optical center y

  Eigen::Matrix3d K;
  K << f_x, 0, c_x,
       0, f_y, c_y,
       0, 0, 1;

  double mu = 0.02;  // typical truncation limit
  double weight = 1.0;  // All depth maps have equal weights

  // Creating the voxel grid
  size_t dimX = 512, dimY = 512, dimZ = 512;
  Eigen::Vector3d gridSize(4, 4, 4); // in meters
  kinect_fusion::VoxelGrid grid(dimX, dimY, dimZ, gridSize);

  // Reposition the voxel grid
  Eigen::Vector3d newCenter(0, 0, 1.16); // Center of the voxel grid
  grid.repositionGrid(newCenter);

  const std::string poseFilePath = "/home/anil/Desktop/kinect_fusion_project/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
  std::string directoryPath = "/home/anil/Desktop/kinect_fusion_project/rgbd_dataset_freiburg1_xyz/depth";

  // Using static functions from utility namespace
  std::vector<std::string> fileNames = kinect_fusion::utility::getPngFilesInDirectory(directoryPath);

  for (const std::string& fileName : fileNames) {
    std::string imagePath = directoryPath + "/" + fileName;
    Eigen::MatrixXd depthImage = kinect_fusion::utility::loadDepthImage(imagePath);

    // Get the pose matching to the current depth map.
    Eigen::Matrix4d pose = kinect_fusion::utility::getPoseFromTimestamp(imagePath, poseFilePath);

    // Update the global TSDF with the current depth map
    grid.updateGlobalTSDF(depthImage, pose, K, mu, weight);
  }

  // Write the resulting TSDF to a file
  kinect_fusion::utility::writeTSDFToFile("global_fusion_result.txt", grid);

  return 0;
}
