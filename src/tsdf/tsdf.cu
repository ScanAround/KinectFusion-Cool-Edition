#include "../frame/Frame.h"
#include "voxel_grid.h"
#include <eigen3/Eigen/Dense>

__global__
void initialize(kinect_fusion::Voxel* grid, size_t dimX, size_t dimY, size_t dimYZ, size_t dimZ, Eigen::Vector3d voxelSize){
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int x = index; x < dimX; x += stride){
    for(size_t y = 0; y < dimY; ++y) {
      for(size_t z = 0; z < dimZ; ++z) {
        grid[x*dimYZ + y*dimZ + z].position = voxelSize.cwiseProduct(Eigen::Vector3d(x, y, z)) + voxelSize * 0.5;
      }
    }
  }
}
namespace kinect_fusion {

VoxelGrid::VoxelGrid(size_t dimX, size_t dimY, size_t dimZ, Eigen::Vector3d gridSize_) : 
                    dimX(dimX), dimY(dimY), dimZ(dimZ), dimYZ(dimY*dimZ), gridSize(gridSize_), 
                    center(gridSize * 0.5) {
  // initially the grid is centered at gridSize * 0.5
  voxelSize = gridSize.cwiseQuotient(Eigen::Vector3d(dimX, dimY, dimZ));
  initializeGrid();
}

void VoxelGrid::initializeGrid() {
  
  cudaMallocManaged((void**)&cu_grid, dimX * dimY * dimZ * sizeof(Voxel));
  initialize <<<1,1024>>> (cu_grid, dimX, dimY, dimZ, dimYZ, gridSize);
  cudaDeviceSynchronize();
}
  
void VoxelGrid::repositionGrid(Eigen::Vector3d newCenter) {
  // Calculate the translation vector
  Eigen::Vector3d translation = newCenter - center;

  for(size_t x = 0; x < dimX; ++x) {
    for(size_t y = 0; y < dimY; ++y) {
      for(size_t z = 0; z < dimZ; ++z) {
        // Apply the translation to each voxel
        grid[x*dimYZ + y*dimZ + z].position += translation;
      }
    }
  }

  center = newCenter; // Update the current center
}

Voxel& VoxelGrid::getVoxel(size_t x, size_t y, size_t z) {
  return grid[x*dimYZ + y*dimZ + z];
}

size_t VoxelGrid::getDimX() const {
  return dimX;
}

size_t VoxelGrid::getDimY() const {
  return dimY;
}

size_t VoxelGrid::getDimZ() const {
  return dimZ;
}

void VoxelGrid::updateGlobalTSDF(const std::vector<Eigen::MatrixXd>& depthMaps, 
                                 const std::vector<Eigen::Matrix4d>& poses,
                                 const std::vector<Eigen::Tensor<double, 3>>& W_R_k,
                                 double mu, 
                                 const Eigen::Matrix3d& K) {
  // For each depth map
  for(int k = 0; k < depthMaps.size(); k++) {
    // For each cell in the TSDF grid
    for(int x = 0; x < dimX; x++) {
      for(int y = 0; y < dimY; y++) {
        for(int z = 0; z < dimZ; z++) {
          Voxel& voxel = getVoxel(x, y, z);
          Eigen::Vector3d p(voxel.position); // The point in the global frame
          Eigen::Vector2d tsdf_result = projectiveTSDF(p, K, poses[k], depthMaps[k], mu);
          double F_R_k_p = tsdf_result[0]; // The TSDF value from the k-th depth map
          double W_R_k_p = W_R_k[k](x, y, z); // The weight from the k-th depth map
          if(std::isnan(F_R_k_p)) continue; // Skip if F_R_k_p is null
          // Update F and W using the provided equations
          if(std::isnan(voxel.tsdfValue) && std::isnan(voxel.weight)){
            voxel.tsdfValue = (W_R_k_p * F_R_k_p) / (W_R_k_p);
          }
          else{
            voxel.tsdfValue = (voxel.weight * voxel.tsdfValue + W_R_k_p * F_R_k_p) / (voxel.weight + W_R_k_p);
          }
          voxel.weight += W_R_k_p;
        }
      }
    }
  }
}

void VoxelGrid::updateGlobalTSDF(Frame& curr_frame,
                                 double mu) {
    //removed weights (setting them to 1) + using frame class now
    for(int x = 0; x < dimX; x++) {
      for(int y = 0; y < dimY; y++) {
        for(int z = 0; z < dimZ; z++) {
          Voxel& voxel = getVoxel(x, y, z);
          Eigen::Vector3d p(voxel.position); // The point in the global frame
          Eigen::Vector2d tsdf_result = projectiveTSDF(p, curr_frame, mu);
          double F_R_k_p = tsdf_result[0]; // The TSDF value from the k-th depth map
          if(std::isnan(F_R_k_p)) continue; // Skip if F_R_k_p is null
          // Update F and W using the provided equations
          if(std::isnan(voxel.tsdfValue)){
            voxel.tsdfValue = F_R_k_p;
          }
          else{
            voxel.tsdfValue = (voxel.tsdfValue + F_R_k_p) / 2;
          }
        }
      }
    }
}

double VoxelGrid::truncatedSignedDistanceFunction(double eta, double mu) {
  if (eta >= -mu)
      return std::min(1.0, -eta / mu);
  else
      return std::numeric_limits<double>::quiet_NaN(); // NaN is used to represent "no data"
}

Eigen::Vector2d VoxelGrid::projectiveTSDF(const Eigen::Vector3d& p, 
                                          const Eigen::Matrix3d& K, 
                                          const Eigen::Matrix4d& T_g_k, 
                                          const Eigen::MatrixXd& R_k, 
                                          double mu) {

  // Transform point p from global frame to the camera coordinate frame at time k
  Eigen::Vector4d p_camera_homogeneous = T_g_k.inverse() * p.homogeneous();

  // p_camera should be a 3D point
  Eigen::Vector3d p_camera = p_camera_homogeneous.head<3>() / p_camera_homogeneous(3);

  // Transform points on the sensor plane into image pixels
  Eigen::Vector3d p_pixel = K * p_camera;

  // Normalize to get image coordinates
  Eigen::Vector2d x = (p_pixel.head<2>() / p_pixel(2));

  // Use floor function to get nearest integer pixel coordinates
  Eigen::Vector2i x_nearest = x.cast<int>();

  // Check if coordinates are within the valid range
  if(x_nearest.x() < 0 || x_nearest.x() >= 640 || x_nearest.y() < 0 || x_nearest.y() >= 480) {
    // If not, return NaN for the TSDF value and the pixel coordinate's norm.
    return Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), x_nearest.norm());
  }

  // Compute lambda
  double lambda = (K.inverse() * x.homogeneous()).norm();

  // Compute eta
  double eta = 1/lambda * p_camera.norm() - R_k(x_nearest.y(), x_nearest.x());
  // double eta = 1/lambda * (T_g_k.block(0,3,3,1) - p_camera).norm() - R_k(x_nearest.y(), x_nearest.x());

  // Compute TSDF value
  double F_R_k_p = truncatedSignedDistanceFunction(eta, mu);

  // Here, we return the TSDF value and the corresponding image coordinate.
  return Eigen::Vector2d(F_R_k_p, x_nearest.norm());
}

Eigen::Vector2d VoxelGrid::projectiveTSDF(const Eigen::Vector3d& p, 
                                          Frame & curr_frame, 
                                          double mu) {
  
  auto K = curr_frame.K_calibration.cast<double>();

  Eigen::Vector2i x = curr_frame.vec_to_pixel(p.cast<float>());

  // Compute lambda
  double lambda = (K.inverse() * x.cast<double>().homogeneous()).norm();

  // Compute eta
  // we have to convert R_k values to meters
  double eta = (1/lambda * (curr_frame.T_gk.block(0,3,3,1).cast<double>() - p).norm()) - curr_frame.get_R(x[0], x[1]) / 5000.0f;

  // Compute TSDF value
  double F_R_k_p = truncatedSignedDistanceFunction(eta, mu);

  // Here, we return the TSDF value and the corresponding image coordinate.
  return Eigen::Vector2d(F_R_k_p, x.norm());
}

}

int main(){

auto start = std::chrono::high_resolution_clock::now();

Eigen::Vector3d gridSize(1,1,1); 
unsigned int res = 128;
kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
std::cout << "time for execution: " << duration << std::endl; 
}