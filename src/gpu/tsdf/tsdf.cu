#include "../../cpu/frame/Frame.h"
#include "../../cpu/tsdf/voxel.h"
#include "../../cpu/tsdf/kinect_fusion_utility.h"
#include "../../cpu/mesher/Marching_Cubes.h"
#include <Eigen/Dense>

__global__ 
void initialize(kinect_fusion::Voxel *cu_grid, int dimX, int dimY, int dimZ, int dimYZ, Eigen::Vector3d voxelSize, dim3 thread_nums, Eigen::Vector3d center){
  int id_x = threadIdx.x + blockIdx.x * thread_nums.x;
  int id_y = threadIdx.y + blockIdx.y * thread_nums.y;
  int id_z = threadIdx.z + blockIdx.z * thread_nums.z;
  if(id_x < dimX && id_y < dimY && id_z < dimZ){
    cu_grid[id_x*dimYZ + id_y*dimZ + id_z].position = voxelSize.cwiseProduct(Eigen::Vector3d(id_x, id_y, id_z)) + voxelSize * 0.5 + center;
  }
}
__device__
double TSDF(double eta, double mu){
  if (eta >= -mu) return min(1.0, -eta / mu);
  return nan("1");
}

__device__ 
Eigen::Vector2i vec_to_pixel(const Eigen::Vector3d vec, Eigen::Matrix3d R_i, Eigen::Vector3d t_i, Eigen::Matrix3d K_calibration, int width, int height){

  Eigen::Vector3d vec_camera_frame = R_i * vec + t_i;
  
  Eigen::Vector3d u_dot = (K_calibration * vec_camera_frame) / vec_camera_frame[2];

  Eigen::Vector2i u;
  if(u_dot[0] >= 0 
  && u_dot[0] <= width 
  && u_dot[1] >= 0 
  && u_dot[1] <= height){
      u << int(u_dot[0]), int(u_dot[1]);
  }
  else{
      u << 0,0 ;
  }
  return u;
}
__device__
double projectiveTSDF(Eigen::Matrix3d K, Eigen::Matrix3d K_i,  Eigen::Vector3d p, Eigen::Matrix3d R_i, Eigen::Vector3d t_i, Eigen::Vector3d t, float *R, int width, int height, double mu){
  Eigen::Vector2i x = vec_to_pixel(p, R_i, t_i, K, width, height);

  // Compute lambda
  double lambda = (K_i * x.cast<double>().homogeneous()).norm();

  // Compute eta
  // we have to convert R_k values to meters
  double eta = (1.0 / lambda) * (t - p).norm() - static_cast<double>((R[x[1]*width + x[0]]) *255.0f* 255.0f) / 5000.0;

  // Compute TSDF value
  double F_R_k_p = TSDF(eta, mu);
  // Here, we return the TSDF value and the corresponding image coordinate.
  return F_R_k_p;
}

__global__
void update(kinect_fusion::Voxel *cu_grid,
            int dimX, int dimY, int dimZ, int dimYZ,
            Eigen::Vector3d voxelSize, Eigen::Matrix3d K, Eigen::Matrix3d K_i, Eigen::Matrix3d R_i, Eigen::Vector3d t_i, Eigen::Vector3d t, 
            float *R, int width, int height, double mu, 
            dim3 thread_nums){
  int id_x = threadIdx.x + blockIdx.x * thread_nums.x;
  int id_y = threadIdx.y + blockIdx.y * thread_nums.y;
  int id_z = threadIdx.z + blockIdx.z * thread_nums.z;
  if(id_x < dimX && id_y < dimY && id_z < dimZ){
    kinect_fusion::Voxel& voxel = cu_grid[id_x*dimYZ + id_y*dimZ + id_z];
    Eigen::Vector3d p(voxel.position); // The point in the global frame
    double F_R = projectiveTSDF(K, K_i, p, R_i, t_i, t, R, width, height, mu);
    if(std::isnan(voxel.tsdfValue)){
      voxel.tsdfValue = F_R;
    }
    else{
      voxel.tsdfValue = (voxel.tsdfValue + F_R) / 2;
    }
  }
}

namespace kinect_fusion {

VoxelGrid::VoxelGrid(size_t dimX, size_t dimY, size_t dimZ, Eigen::Vector3d gridSize_, Eigen::Vector3d ctr_of_mass) : 
                    dimX(dimX), dimY(dimY), dimZ(dimZ), dimYZ(dimY*dimZ), gridSize(gridSize_), 
                    center(-0.5 * gridSize_ + ctr_of_mass) {
                    // center(-0.5*gridSize_){
  grid.resize(dimX * dimYZ);
  voxelSize = gridSize.cwiseQuotient(Eigen::Vector3d(dimX, dimY, dimZ));
  initializeGrid();
}

void VoxelGrid::initializeGrid() {
  cudaError_t cudaStatus = cudaMallocManaged(&cu_grid, dimX * dimYZ * sizeof(Voxel));
  if(cudaStatus != cudaSuccess){
    std::cout << "Problem in CudaMalloc: " << cudaGetErrorString(cudaStatus) << std::endl;
  }
  const int tile_dim = 4; // make sure it's a multiple of dimX -> not sure what the optimal tile_dim is 
  dim3 thread_nums(tile_dim, tile_dim, tile_dim);  
  dim3 block_nums(dimX/tile_dim, dimY/tile_dim, dimZ/tile_dim);

  initialize <<<block_nums,thread_nums>>> (cu_grid, dimX, dimY, dimZ, dimYZ, voxelSize, thread_nums, center);
  cudaDeviceSynchronize();
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

void VoxelGrid::updateGlobalTSDF(Frame& curr_frame,
                                 double mu) {

  float *R;
  auto CudaAssignemnt = cudaMalloc(&R, sizeof(float) * curr_frame.width * curr_frame.height);
  if(CudaAssignemnt != cudaSuccess){
    std::cout << "Problem in Assignment: " << CudaAssignemnt <<std::endl;
  }
  auto CudaCopy = cudaMemcpy(R, curr_frame.Raw_k, sizeof(float) * curr_frame.width * curr_frame.height, cudaMemcpyHostToDevice);
  if(CudaCopy != cudaSuccess){
    std::cout << "Problem in Copying: " << CudaCopy <<std::endl;
  }

  const int tile_dim = 4; // make sure it's a multiple of dimX -> not sure what the optimal tile_dim is 
  dim3 thread_nums(tile_dim, tile_dim, tile_dim);  // maybe make this a class attribute
  dim3 block_nums(dimX/tile_dim, dimY/tile_dim, dimZ/tile_dim);
  
  auto K = curr_frame.K_calibration.cast<double>();
  auto K_i = K.inverse();
  auto T_gk = curr_frame.T_gk.cast<double>();
  auto R_i = T_gk.inverse().block(0,0,3,3);
  auto t_i = T_gk.inverse().block(0,3,3,1);
  auto t = T_gk.block(0,3,3,1);
  update <<<block_nums,thread_nums>>> (cu_grid, dimX, dimY, dimZ, dimYZ, 
                                       voxelSize, K , K_i, R_i, t_i, t,
                                       R, curr_frame.width, curr_frame.height, mu, thread_nums);
  cudaDeviceSynchronize();
  
  cudaError_t cudaStatus2 = cudaMemcpy(grid.data(), cu_grid, dimX * dimYZ * sizeof(Voxel), cudaMemcpyDeviceToHost);
  if(cudaStatus2 != cudaSuccess){
    std::cout << "Problem in Copying: " << cudaGetErrorString(cudaStatus2) << std::endl;
  };
  cudaFree(R);
}

}


// int main(){

// auto pose_f = Eigen::Matrix4f::Identity();

// const char* img_loc = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"; 

// Frame* frame1 = new Frame(img_loc, pose_f, 1.0);

// frame1 -> process_image();

// frame1 -> save_G_off_format("G.obj");
// frame1 -> save_off_format("original.obj");

// Eigen::Vector3d gridSize(4,4,4); 
// unsigned int res = 256;
// Eigen::Vector3d ctr_of_mass = (frame1 -> center_of_mass).cast<double>();
// kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize, ctr_of_mass);
// double mu = 0.02;
// auto start = std::chrono::high_resolution_clock::now();
// grid.updateGlobalTSDF(*frame1, mu);

// auto end = std::chrono::high_resolution_clock::now();
// auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

// // kinect_fusion::utility::writeTSDFToFile("TSDF.txt", grid);
// Marching_Cubes::Mesher(grid, 0, "mesh.off");

// std::cout << "time for execution: " << duration << std::endl; 
// }