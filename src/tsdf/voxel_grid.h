#pragma once

#include "voxel.h"
#include <vector>      // for std::vector
#include <eigen3/Eigen/Dense> // for Eigen::Vector3d
#include <eigen3/unsupported/Eigen/CXX11/Tensor> // for Eigen::Tensor<double, 3>
#include "../frame/Frame.h"
// TO DO: Implement the real-time version of the function above in which we process each incoming 
// depth map as it arrives, updating the global TSDF and weights incrementally instead of 
// remembering all of the depthmaps, poses, R_k, W_R_k etc. starting from the very first one to 
// the current one. 

namespace kinect_fusion {
// Updated VoxelGrid
class VoxelGrid {
private:
  std::vector<std::vector<std::vector<Voxel>>> grid;
  size_t dimX, dimY, dimZ;
  Eigen::Vector3d gridSize;
  Eigen::Vector3d voxelSize;
  Eigen::Vector3d center;

  // Psi function, which does the truncation withing the projective TSDF, as described in the 
  // research paper 
  // "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/ismar_2011.pdf".
  double truncatedSignedDistanceFunction(double eta, double mu);

public:

  // Let's say that dimX, dimY and dimZ are 512 and the grid size is 4m x 4m x 4m. The voxel size 
  // is calculated as follows:
  // gridSize is a vector representing the overall size of the grid in each dimension, which in 
  // this case is (4, 4, 4).

  // Eigen::Vector3d(dimX, dimY, dimZ) creates a vector representing the number of voxels in each 
  // dimension, which is (512, 512, 512) in this case.
  // Then, the cwiseQuotient function performs element-wise division of the grid size by the 
  // number of voxels. This gives the size of each voxel.

  // In our case, the size of each voxel in each dimension is calculated as follows:
  // Voxel size along X: 4 / 512 = 0.0078125 meters
  // Voxel size along Y: 4 / 512 = 0.0078125 meters
  // Voxel size along Z: 4 / 512 = 0.0078125 meters
  // So, each voxel in our grid would be a cube with a side length of approximately 0.0078125 
  // meters.
  VoxelGrid(size_t dimX, size_t dimY, size_t dimZ, Eigen::Vector3d gridSize_);

  // The grid is initialized with the origin at [0,0,0] and center at gridSize * 0.5.
  void initializeGrid();
  
  // This function moves the voxel grid to a new center point. It can be called as 
  // "grid.repositionGrid(Eigen::Vector3d(0, 0, 1.16));". This repositions the grid so that its 
  // center is at [0, 0, 1.16]. The repositionGrid function calculates the translation vector as 
  // the difference between the new center and the current center, then applies this translation 
  // to each voxel in the grid. This effectively moves the entire grid to the new position.
  void repositionGrid(Eigen::Vector3d newCenter);

  Voxel& getVoxel(size_t x, size_t y, size_t z);

  size_t getDimX() const;

  size_t getDimY() const;

  size_t getDimZ() const;

  // Global fusion of all depth maps in the volume to a single TSDF as described in the research
  // paper "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/ismar_2011.pdf".
  void updateGlobalTSDF(const std::vector<Eigen::MatrixXd>& depthMaps, 
                        const std::vector<Eigen::Matrix4d>& poses,
                        const std::vector<Eigen::Tensor<double, 3>>& W_R_k,
                        double mu, 
                        const Eigen::Matrix3d& K);
  
  void updateGlobalTSDF(Frame& curr_frame,
                        double mu);

  // Projective TSDF function as described in the research paper 
  // "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/ismar_2011.pdf".
  Eigen::Vector2d projectiveTSDF(const Eigen::Vector3d& p, 
                                 const Eigen::Matrix3d& K, 
                                 const Eigen::Matrix4d& T_g_k, 
                                 const Eigen::MatrixXd& R_k, 
                                 double mu);
  

  Eigen::Vector2d projectiveTSDF(const Eigen::Vector3d& p, 
                               Frame & curr_frame, 
                               double mu);
};

}