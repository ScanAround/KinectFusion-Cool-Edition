#include "cpu/frame/Frame_Pyramid.h"
#include "cpu/icp/ICP.h"
#include "cpu/tsdf/voxel_grid.h"
#include "cpu/tsdf/kinect_fusion_utility.h"
#include "cpu/raytracing/Raycasting.h"
#include "cpu/mesher/Marching_Cubes.h"
#include "chrono"

int main(){

        
    FreeImage_Initialise();
    
    //initiating mesher
    std::unique_ptr<Marching_Cubes> mesher = std::make_unique<Marching_Cubes>();
    
    std::string s_dir = "data/rgbd_dataset_freiburg1_xyz/depth"; 
    auto filenames = kinect_fusion::utility::getPngFilesInDirectory(s_dir);
    
    Frame_Pyramid curr_frame(s_dir + "/" + filenames[0]);
    curr_frame.Depth_Pyramid[0]->save_G_off_format("outputs/point_clouds/G.obj");
    
    //initiating grid
    Eigen::Vector3d gridSize(4,4,4); 
    unsigned int res = 256;
    kinect_fusion::VoxelGrid grid(res ,res ,res ,gridSize, curr_frame.Depth_Pyramid[0]->center_of_mass.cast<double>());
    float mu = 0.1;
    // float mu = 3.0;
    
    //somehow we're getting a problem because of our initial T_gk probably
    grid.updateGlobalTSDF(*curr_frame.Depth_Pyramid[0], mu);
    

    mesher -> Mesher(grid, "outputs/meshes/mesh1.off");
    auto T = curr_frame.T_gk;

    for(int file_idx = 0; file_idx < filenames.size()-1; ++file_idx){
        auto start = std::chrono::high_resolution_clock::now();
        Raycasting prev_r(grid, T.block(0,0,3,3), T.block(0,3,3,1));
        prev_r.castAllCuda();

        auto raycast_end = std::chrono::high_resolution_clock::now();

        Frame_Pyramid prev_frame(prev_r.getVertices(), prev_r.getNormals(), T);
        prev_frame.set_T_gk(T);

        auto frame_end = std::chrono::high_resolution_clock::now();

        Frame_Pyramid curr_frame_(s_dir + "/" + filenames[file_idx + 1]);
        curr_frame_.set_T_gk(T); // done so converging is faster (theoretically + still testing)
        
        ICP icp(curr_frame_, prev_frame, 0.045f, 0.95f);
        
        auto icp_end = std::chrono::high_resolution_clock::now();
        
        T = icp.pyramid_ICP(false);

        grid.updateGlobalTSDF(*curr_frame_.Depth_Pyramid[0], mu);
        
        auto tsdf_end = std::chrono::high_resolution_clock::now();

        auto raycast_duration = std::chrono::duration_cast<std::chrono::milliseconds>(raycast_end - start);
        auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - raycast_end);
        auto icp_duration = std::chrono::duration_cast<std::chrono::milliseconds>(icp_end - frame_end);
        auto tsdf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tsdf_end - icp_end);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(icp_end - start);
        
        std::cout << "raycast time: " << raycast_duration.count() << std::endl;
        std::cout << "frame time: " << frame_duration.count() << std::endl;
        std::cout << "icp time: " << icp_duration.count() << std::endl;
        std::cout << "tsdf time: " << tsdf_duration.count() << std::endl;
        std::cout << "total time: " << total_duration.count() << std::endl;

        if(file_idx != 0 && file_idx % 200 == 0) {
            mesher -> Mesher(grid, "outputs/meshes/mesh" + std::to_string(file_idx) + ".off");
        }
    }

    FreeImage_DeInitialise();
}