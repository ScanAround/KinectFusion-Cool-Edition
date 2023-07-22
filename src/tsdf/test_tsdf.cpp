#include "voxel_grid.h"
#include "kinect_fusion_utility.h"
#include "../mesher/Marching_Cubes.h"
#include <algorithm>
#include <chrono>

int main() {

auto start = std::chrono::high_resolution_clock::now();

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

Eigen::Vector3d gridSize(1,1,1); 
unsigned int res = 128;

kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize);
// grid.repositionGrid(Eigen::Vector3d(4,4,4)); // doesn't do anything actually!!!

std::unique_ptr<Marching_Cubes> mesher = std::make_unique<Marching_Cubes>();

double mu = 0.02;

grid.updateGlobalTSDF(*frame1, mu);

// Write the resulting TSDF to a file
kinect_fusion::utility::writeTSDFToFile("global_fusion_result.txt", grid);

mesher -> Mesher(grid, 0, "mesh2.off");

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

std::cout << "time for execution: " << duration << std::endl; 
}