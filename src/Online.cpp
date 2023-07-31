#include "cpu/frame/Frame_Pyramid.h"
#include "cpu/icp/ICP.h"
#include "cpu/tsdf/voxel_grid.h"
#include "cpu/tsdf/kinect_fusion_utility.h"
#include "cpu/raytracing/Raycasting.h"
#include "cpu/mesher/Marching_Cubes.h"
#include "cpu/frame/RealSense.h"

int main(){

    FreeImage_Initialise();
    
    //initiating mesher
    std::unique_ptr<Marching_Cubes> mesher = std::make_unique<Marching_Cubes>();
    std::unique_ptr<RealSensor> our_sensor = std::make_unique<RealSensor>();

    our_sensor -> start();
    Eigen::Matrix3f K = our_sensor -> getDepthIntrinsics();

    Frame_Pyramid curr_frame(our_sensor -> processNextFrame(), K);
    curr_frame.Depth_Pyramid[0]->save_G_off_format("outputs/point_clouds/G.obj");
    
    //initiating grid
    Eigen::Vector3d gridSize(4,4,4); 
    unsigned int res = 256;
    kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize, curr_frame.Depth_Pyramid[0]->center_of_mass.cast<double>());
    float mu = 0.02;
    
    //somehow we're getting a problem because of our initial T_gk probably
    grid.updateGlobalTSDF(*curr_frame.Depth_Pyramid[0], mu);
    mesher -> Mesher(grid, "outputs/meshes/mesh_1.off");
    auto T = curr_frame.T_gk;

    for(int file_idx = 0; file_idx < 100; ++file_idx){ 
        Raycasting prev_r(grid, T.block(0,0,3,3), T.block(0,3,3,1));
        prev_r.castAllCuda();
        Frame_Pyramid prev_frame(prev_r.getVertices(), prev_r.getNormals(), T);
        
        // Frame_Pyramid prev_frame(our_sensor -> processNextFrame(), K, T);
        // prev_frame.set_T_gk(T);

        // prev_frame.Depth_Pyramid[0]->save_G_off_format("outputs/point_clouds/pc_G_previous" + std::to_string(file_idx) + ".obj");

        Frame_Pyramid curr_frame_(our_sensor -> processNextFrame(), K, T);
        curr_frame_.set_T_gk(T); // done so converging is faster (theoretically + still testing)
        
        ICP icp(curr_frame_, prev_frame, 0.1f, 0.7f);
        T = icp.pyramid_ICP(false);

        grid.updateGlobalTSDF(*curr_frame_.Depth_Pyramid[0], mu);
        // curr_frame_.Depth_Pyramid[0]->save_off_format("outputs/point_clouds/pc" +std::to_string(file_idx) + ".obj");
        // curr_frame_.Depth_Pyramid[0]->save_G_off_format("outputs/point_clouds/pc_G" +std::to_string(file_idx) + ".obj");

        if(file_idx != 0 && file_idx % 5 == 0) mesher -> Mesher(grid, "outputs/meshes/mesh" + std::to_string(file_idx) + ".off");
    }

    FreeImage_DeInitialise();
}