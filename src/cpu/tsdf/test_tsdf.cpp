#include "voxel_grid.h"
#include "../raytracing/Raycasting.h"
#include "../tsdf/kinect_fusion_utility.h"
#include "../mesher/Marching_Cubes.h"
#include <algorithm>
#include <chrono>

// TODO: this can be moved to Raycasting if you want
void writePointCloud(const std::string& filename, const std::vector<Eigen::Vector3f>& _vertices, const std::vector<Eigen::Vector3f>& _normals)
{
	std::ofstream file(filename);

	for (unsigned int i = 0; i < _vertices.size(); ++i)
	{
		file << "v " << _vertices[i][0] << " " << _vertices[i][1] << " " << _vertices[i][2] << std::endl;
		file << "vn " << _normals[i][0] << " " << _normals[i][1] << " " << _normals[i][2] << std::endl;
	}
}

int main() {

// auto start = std::chrono::high_resolution_clock::now();

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

const char* img_loc = "C:\\Users\\marcw\\Desktop\\BMC_TUM\\sose23\\3dsmc\\Exercises\\Data\\rgbd_dataset_freiburg1_xyz\\depth\\1305031102.160407.png"; 

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

// std::vector<Eigen::Vector3f> vertices;
// std::vector<Eigen::Vector3f> normals;

Eigen::Matrix3f rotation; 
rotation << float(pose(0, 0)), float(pose(0, 1)), float(pose(0, 2)),
            float(pose(1, 0)), float(pose(1, 1)), float(pose(1, 2)),
            float(pose(2, 0)), float(pose(2, 1)), float(pose(2, 2));

Eigen::Vector3f translation(1.3434f, 0.6271f, 1.6606f);

auto start = std::chrono::high_resolution_clock::now();
Raycasting r(grid, rotation, translation);
	
r.castAllCuda();

r.writePointCloud("pointcloud.obj");

r.freeCuda();

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

mesher -> Mesher(grid, 0, "mesh2.off");

kinect_fusion::utility::writeTSDFToFile("tsdf_cuda.txt", grid);

// auto end = std::chrono::high_resolution_clock::now();
// auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

std::cout << "time for execution: " << duration << std::endl; 
}