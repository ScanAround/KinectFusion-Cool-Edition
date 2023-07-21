#include "Lookup_Tables.h"
#include "Marching_Cubes.h"
#include "../tsdf/voxel_grid.h"
#include "../tsdf/voxel.h"
#include <eigen3/Eigen/Dense> 
#include "../tsdf/kinect_fusion_utility.h"

#define TRUNCATION 1.0

int main(){
    Eigen::Vector3d gridSize(1.1, 1.1, 1.1); 
    unsigned int res = 450;
    std::vector<Eigen::MatrixXd> depthMaps;
    std::vector<Eigen::Matrix4d> poses;
    std::vector<Eigen::Tensor<double, 3>> W_R_k;

    kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize);
    std::unique_ptr<Marching_Cubes> mesher = std::make_unique<Marching_Cubes>();
    
    Eigen::MatrixXd depthImage = kinect_fusion::utility::loadDepthImage("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/src/mesher/sphere_depth_map.png");
    depthMaps.push_back(depthImage);

    double tx = 1.3226, ty = 0.6209, tz = 1.6406;
    double qx = 0.6525, qy = 0.6373, qz = -0.2971, qw = -0.2825;

    Eigen::Matrix4d pose;
    Eigen::Vector3d trans(tx, ty, tz);
    Eigen::Quaterniond quat(qw, qx, qy, qz);
    
    pose.topLeftCorner<3, 3>() = quat.toRotationMatrix();
    pose.topRightCorner<3, 1>() = trans;
    pose(3,0) = pose(3,1) = pose(3,2) = 0.0;
    pose(3,3) = 1.0;
    poses.push_back(pose);

    double f_x = 517.3;  // focal length x
    double f_y = 516.5;  // focal length y
    double c_x = 318.6;  // optical center x
    double c_y = 255.3;  // optical center y

    Eigen::Matrix3d K;
    K << f_x, 0, c_x,
        0, f_y, c_y,
        0, 0, 1;
    
    double mu = 0.02;

    Eigen::Tensor<double, 3> weights(res, res, res);
    weights.setConstant(1.0);
    W_R_k.push_back(weights);

    grid.updateGlobalTSDF(depthMaps, poses, W_R_k, mu, K);

    // Write the resulting TSDF to a file
    kinect_fusion::utility::writeTSDFToFile("global_fusion_result.txt", grid);

    mesher -> Mesher(grid);
}