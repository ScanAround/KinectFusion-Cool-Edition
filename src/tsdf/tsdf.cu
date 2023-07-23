#include "../frame/Frame.h"
#include "voxel_grid.h"
#include "../mesher/Marching_Cubes.h"
#include <eigen3/Eigen/Dense>
#define CU_NAN              __longlong_as_double(0xfff8000000000000ULL)

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
__device__
double TSDF(double eta, double mu) {
  if (eta >= -mu)
      return min(1.0, -eta / mu);
  else
      return CU_NAN; // NaN is used to represent "no data"
}
__device__
Eigen::Vector2i cu_vec_to_pixel(const Eigen::Vector3d& vec, Eigen::Matrix4d T_gk, Eigen::Matrix3d K, int width, int height){
        Eigen::Matrix3d rotation = T_gk.inverse().block(0,0,3,3);
        Eigen::Vector3d translation = T_gk.inverse().block(0,3,3,1);
        
        Eigen::Vector3d vec_camera_frame = rotation * vec + translation;
        
        Eigen::Vector3d u_dot = (K * vec_camera_frame) / vec_camera_frame[2];

        Eigen::Vector2i u;
        if(u_dot[0] >= 0 
        && u_dot[0] <= width 
        && u_dot[1] >= 0 
        && u_dot[1] <= height){
            // making sure u is within the image we have 
            u << int(u_dot[0]), int(u_dot[1]);
        }
        else{
            u << 0,0 ;
        }
        return u;
}
__device__
Eigen::Vector2d proj_TSDF(const Eigen::Vector3d& p, 
                          Frame& curr_frame, 
                          double mu,
                          float * R) {
  
  auto K = curr_frame.K_calibration.cast<double>();
  auto T_gk = curr_frame.T_gk.cast<double>();
  auto width = curr_frame.width;
  auto height = curr_frame.height;

  Eigen::Vector2i x = cu_vec_to_pixel(p, T_gk, K, width, height);

  // Compute lambda
  double lambda = (K.inverse() * x.cast<double>().homogeneous()).norm();

  // Compute eta
  // we have to convert R_k values to meters
  double eta = (1/lambda * (curr_frame.T_gk.block(0,3,3,1).cast<double>() - p).norm()) - R[x[1]*width + x[0]] * 255.0f * 255.0f / 5000.0f;

  // Compute TSDF value
  double F_R_k_p = TSDF(eta, mu);

  // Here, we return the TSDF value and the corresponding image coordinate.
  return Eigen::Vector2d(F_R_k_p, x.norm());
}

__global__
void update(kinect_fusion::Voxel* grid, Frame& curr_frame, double mu, size_t dimX, size_t dimY, size_t dimYZ, size_t dimZ, float * R){
int index = threadIdx.x;
int stride = blockDim.x;
for(int x = index; x < dimX; x+=stride) {
      for(int y = 0; y < dimY; y++) {
        for(int z = 0; z < dimZ; z++) {
          kinect_fusion::Voxel& voxel = grid[x*dimYZ + y*dimZ + z];
          Eigen::Vector3d p(voxel.position); // The point in the global frame
          Eigen::Vector2d tsdf_result = proj_TSDF(p, curr_frame, mu, R);
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
  initialize <<<1,512>>> (cu_grid, dimX, dimY, dimZ, dimYZ, gridSize);
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

  float * R;
  cudaMalloc((void**)&R, sizeof(float) * curr_frame.get_R_size());
  cudaMemcpy(R, &curr_frame.Raw_k, sizeof(float) * curr_frame.get_R_size(), cudaMemcpyHostToDevice);

  update <<<1,512>>> (cu_grid, curr_frame, mu, dimX, dimY, dimYZ, dimZ, R);
  cudaDeviceSynchronize();
  
  cudaMemcpy(grid.data(), &cu_grid, dimX * dimY * dimZ * sizeof(Voxel), cudaMemcpyDeviceToHost);
  // cudaFree(cu_grid);
  // cudaFree(R);
}

}

int main(){


double tx = 1.3434, ty = 0.6271, tz = 1.6606;
double qx = 0.6583, qy = 0.6112, qz = -0.2938, qw = -0.3266;

// Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
Eigen::Matrix4d pose;
Eigen::Vector3d trans(tx, ty, tz);
Eigen::Quaterniond quat(qw, qx, qy, qz);

pose.topLeftCorner<3, 3>() = quat.toRotationMatrix();
pose.topRightCorner<3, 1>() = trans;
pose(3,0) = pose(3,1) = pose(3,2) = 0.0;
pose(3,3) = 1.0;

auto pose_f = pose.cast<float>();

const char* img_loc = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"; 

Frame* frame1 = new Frame(img_loc, pose_f, 1.0);

frame1 -> process_image();

std::vector<Eigen::Vector3f> V_tk;

frame1 -> apply_transform(pose_f, V_tk);

std::ofstream OffFile("G_Frame1.obj");
for(auto V : V_tk){
    OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
}

auto start = std::chrono::high_resolution_clock::now();

Eigen::Vector3d gridSize(1,1,1); 
unsigned int res = 128;

kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize);

double mu = 0.02;

grid.updateGlobalTSDF(*frame1, mu);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

std::cout << "time for execution: " << duration << std::endl; 

Marching_Cubes::Mesher(grid, 0, "mesh2.off");
}