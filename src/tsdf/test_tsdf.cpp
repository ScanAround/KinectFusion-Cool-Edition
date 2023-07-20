#include "voxel_grid.h"
#include "kinect_fusion_utility.h"

int main() {

  try {
    double f_x = 517.3;  // focal length x
    double f_y = 516.5;  // focal length y
    double c_x = 318.6;  // optical center x
    double c_y = 255.3;  // optical center y

    Eigen::Matrix3d K;
    K << f_x, 0, c_x,
        0, f_y, c_y,
        0, 0, 1;

    double mu = 0.02;  // typical truncation limit

    // Creating the voxel grid
    size_t dimX = 512, dimY = 512, dimZ = 512;
    Eigen::Vector3d gridSize(4, 4, 4); // in meters
    kinect_fusion::VoxelGrid grid(dimX, dimY, dimZ, gridSize);

    // Reposition the voxel grid
    Eigen::Vector3d newCenter(0, 0, 1.16); // Center of the voxel grid
    grid.repositionGrid(newCenter);

    const std::string poseFilePath = "../../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
    std::string directoryPath = "../../data/rgbd_dataset_freiburg1_xyz/depth";

    // Using static functions from utility namespace
    std::vector<std::string> fileNames = kinect_fusion::utility::getPngFilesInDirectory(directoryPath);

    std::vector<Eigen::MatrixXd> depthMaps;
    std::vector<Eigen::Matrix4d> poses;
    std::vector<Eigen::Tensor<double, 3>> W_R_k;
    size_t counter {0};
    for (const std::string& fileName : fileNames) {
      std::string imagePath = directoryPath + "/" + fileName;
      Eigen::MatrixXd depthImage = kinect_fusion::utility::loadDepthImage(imagePath);
      depthMaps.push_back(depthImage);

      std::cout << "Depth map" << counter << " extracted." << std::endl;

      // All depth maps have equal weights
      // Assuming all weights are 1.0 initially
      Eigen::Tensor<double, 3> weights(dimX, dimY, dimZ);
      weights.setConstant(1.0);
      W_R_k.push_back(weights);

      // Get the pose matching to the current depth map.
      Eigen::Matrix4d pose = kinect_fusion::utility::getPoseFromTimestamp(imagePath, poseFilePath);
      poses.push_back(pose);

      // Update the global TSDF with the current depth map
      grid.updateGlobalTSDF(depthMaps, poses, W_R_k, mu, K);
      
      counter++;
    }

    // Write the resulting TSDF to a file
    kinect_fusion::utility::writeTSDFToFile("global_fusion_result.txt", grid);
  }
  catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    return 1;
  }
  catch (...) {
    std::cerr << "Unknown exception caught\n";
    return 1;
  }

  return 0;
}