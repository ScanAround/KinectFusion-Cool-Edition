#include "../../cpu/frame/Frame.h"
#include "../../cpu/tsdf/voxel.h"
#include "../../cpu/mesher/Marching_Cubes.h"
#include <eigen3/Eigen/Dense>
#define CU_NAN              __longlong_as_double(0xfff8000000000000ULL)

__global__ 
void initialize(kinect_fusion::Voxel *cu_grid, int dimX, int dimY, int dimZ, int dimYZ, Eigen::Vector3d voxelSize, dim3 thread_nums){
int id_x = threadIdx.x + blockIdx.x * thread_nums.x;
int id_y = threadIdx.y + blockIdx.y * thread_nums.y;
int id_z = threadIdx.z + blockIdx.z * thread_nums.z;
if(id_x < dimX && id_y < dimY && id_z < dimZ){
  cu_grid[id_x*dimYZ + id_y*dimZ + id_z].position = voxelSize.cwiseProduct(Eigen::Vector3d(id_x, id_y, id_z)) + voxelSize * 0.5;
}
}
namespace kinect_fusion {

VoxelGrid::VoxelGrid(size_t dimX, size_t dimY, size_t dimZ, Eigen::Vector3d gridSize_) : 
                    dimX(dimX), dimY(dimY), dimZ(dimZ), dimYZ(dimY*dimZ), gridSize(gridSize_), 
                    center(gridSize * 0.5) {
  // initially the grid is centered at gridSize * 0.5
  // grid.resize(dimX * dimYZ);
  grid.resize(dimX * dimYZ);
  voxelSize = gridSize.cwiseQuotient(Eigen::Vector3d(dimX, dimY, dimZ));
  initializeGrid();
}

void VoxelGrid::initializeGrid() {
  cudaError_t cudaStatus = cudaMallocManaged(&cu_grid, dimX * dimYZ * sizeof(Voxel));
  if(cudaStatus != cudaSuccess){
    std::cout << "Problem in CudaMalloc: " << cudaGetErrorString(cudaStatus) << std::endl;
  }
  const int tile_dim = 4; // make sure it's a multiple of dimX
  dim3 thread_nums(tile_dim, tile_dim, tile_dim);  
  dim3 block_nums(dimX/tile_dim, dimY/tile_dim, dimZ/tile_dim);

  initialize <<<block_nums,thread_nums>>> (cu_grid, dimX, dimY, dimZ, dimYZ, voxelSize, thread_nums);
  std::cout << cudaGetLastError() <<std::endl;
  cudaDeviceSynchronize();

  cudaError_t cudaStatus2 = cudaMemcpy(grid.data(), cu_grid, dimX * dimYZ * sizeof(Voxel), cudaMemcpyDeviceToHost);
  if(cudaStatus2 != cudaSuccess){
    std::cout << "Problem in Copying: " << cudaGetErrorString(cudaStatus2) << std::endl;
  };
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

  // float * R;
  // cudaMalloc((void**)&R, sizeof(float) * curr_frame.get_R_size());
  // cudaMemcpy(R, &curr_frame.Raw_k, sizeof(float) * curr_frame.get_R_size(), cudaMemcpyHostToDevice);

  // update <<<1,512>>> (cu_grid, curr_frame, mu, dimX, dimY, dimYZ, dimZ, R);
  // cudaDeviceSynchronize();
  
  // cudaMemcpy(grid.data(), &cu_grid, dimX * dimY * dimZ * sizeof(Voxel), cudaMemcpyDeviceToHost);
  // // cudaFree(cu_grid);
  // // cudaFree(R);
}

}

int main(){


// double tx = 1.3434, ty = 0.6271, tz = 1.6606;
// double qx = 0.6583, qy = 0.6112, qz = -0.2938, qw = -0.3266;

// // Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
// Eigen::Matrix4d pose;
// Eigen::Vector3d trans(tx, ty, tz);
// Eigen::Quaterniond quat(qw, qx, qy, qz);

// pose.topLeftCorner<3, 3>() = quat.toRotationMatrix();
// pose.topRightCorner<3, 1>() = trans;
// pose(3,0) = pose(3,1) = pose(3,2) = 0.0;
// pose(3,3) = 1.0;

// auto pose_f = pose.cast<float>();

// const char* img_loc = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"; 

// Frame* frame1 = new Frame(img_loc, pose_f, 1.0);

// frame1 -> process_image();

// std::vector<Eigen::Vector3f> V_tk;

// frame1 -> apply_transform(pose_f, V_tk);

// std::ofstream OffFile("G_Frame1.obj");
// for(auto V : V_tk){
//     OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
// }

auto start = std::chrono::high_resolution_clock::now();

Eigen::Vector3d gridSize(1,1,1); 
unsigned int res = 128;

kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize);
// double mu = 0.02;
// grid.updateGlobalTSDF(*frame1, mu);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

std::cout << "time for execution: " << duration << std::endl; 
std::cout << grid.getVoxel(1,1,1).position << std::endl;
}