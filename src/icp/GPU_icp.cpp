#include "GPU_icp.h"
#include "../frame/Frame_Pyramid.h"
#include <iostream>
#include <fstream>
#include <FreeImage.h>
#define MINF -std::numeric_limits<float>::infinity()

void ICP::correspondence_finder(Eigen::Matrix4f T_curr_frame, Frame & curr_frame, Frame & prev_frame, std::vector<std::pair<int, int>>& matches){
    // will return the nearest neighbour index for each source vector i.e.
    // we don't take normals into consideration
    // 0 1 2 3 4 5 source vector indices
    // 5 2 3 1 4 4 prev_frame vector nearest neighbor to source indices
    
    std::vector<Eigen::Vector3f> curr_V_k_transformed; //transformed V
    std::vector<Eigen::Vector3f> curr_N_k_transformed; //transformed V

    curr_frame.apply_transform(T_curr_frame, curr_V_k_transformed, curr_N_k_transformed); //transforming current frame's V_k -> V_tk according to previous frame

    std::cout << "Finding Correspondences" << std::endl;

    for(int i = 0; i < curr_V_k_transformed.size(); i++){
        if(!std::isnan(curr_V_k_transformed[i][0])){
            Eigen::Vector2i pixel = prev_frame.vec_to_pixel(curr_V_k_transformed[i]);
            int idx_in_prev = pixel[1]*prev_frame.width + pixel[0];
            
            //Normals aren't calculated in the last row => check if idx is less than last row idxs
            if(idx_in_prev < (prev_frame.height-1) * prev_frame.width){
                //check if idx_in_prev is a valid vertex in previous frame
                if(!std::isnan(prev_frame.N_gk[idx_in_prev][0])){
                    // check if distances between corresponding vectors are below threshold
                    if((curr_V_k_transformed[i] - prev_frame.V_gk[idx_in_prev]).norm() <= this->distance_threshold){
                        // check if distances between corresponding angles are below threshold
                        if((curr_N_k_transformed[i].dot(prev_frame.N_gk[idx_in_prev])) <= this->angle_threshold){
                            matches.push_back(std::make_pair(i, idx_in_prev));
                        }
                    }
                }
            }
        }
    }
}

Eigen::Matrix4f ICP::point_to_plane_solver(Frame & curr_frame, Frame & prev_frame, int iterations, bool cuda){
    
    // source is the live frame F_k and the prev_frame is the ray-casted previous frame F_k-1
    Eigen::Matrix4f T_gk_z = curr_frame.T_gk;

    std::vector<Eigen::Vector3f> source_vectors = curr_frame.V_k; 

    if(!cuda){
        //for loop since first without parallelization
        for(int i = 0; i < iterations; i++){
            
            prev_frame.apply_G_transform();

            std::vector<std::pair<int, int>> correspondences;

            this->correspondence_finder(T_gk_z, curr_frame, prev_frame, correspondences);
            
            int corr_size = correspondences.size();
            
            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6); // supposed to be ATA of the system according to paper

            Eigen::MatrixXf A_jT = Eigen::MatrixXf::Zero(6, 1);
            
            Eigen::MatrixXf b = Eigen::MatrixXf::Zero(6, 1); // supposed to be ATb according to paper
            
            // for(auto i: source.M_k1){
            for(int j = 0; j < corr_size; ++j){
                // get each point's contribution to ATA
                // we have to transform our source.V_k into our estimated transformation matrix
                Eigen::Vector3f s_i = T_gk_z.block(0,0,3,3) * curr_frame.V_k[correspondences[j].first] + T_gk_z.block(0,3,3,1);
                Eigen::Vector3f d_i = prev_frame.V_gk[correspondences[j].second];
                
                Eigen::Vector3f n_i = prev_frame.N_gk[correspondences[j].second];
                
                A_jT.block(0,0,3,1) = s_i.cross(n_i);
                A_jT.block(3,0,3,1) = n_i;

                A.selfadjointView<Eigen::Lower>().rankUpdate(A_jT); // only calculates the lower triangle (eigen's ldlt calculates with lower triangle) and adds

                b += A_jT * (n_i.dot(d_i) - n_i.dot(s_i));
                
            }
            // std::cout << A << std::endl;
            // std::cout << b << std::endl;

            Eigen::Vector<float, 6> x = A.ldlt().solve(b); //ldlt because ATA not always Positive Definite
            
            float alpha = x[0];
            float beta = x[1];
            float gamma = x[2];


            T_gk_z <<      1 ,  alpha*beta - gamma , alpha*gamma + beta , x[3],
                       gamma ,  alpha*beta*gamma + 1   ,  beta*gamma - alpha , x[4],
                       -beta , alpha ,   1   , x[5],
                        0    ,  0    ,   0   ,  1  ; 
            
            
        }
        
    }

    else{
        //parallelization part
    }
    
    return T_gk_z;
    
    }

Eigen::Matrix4f ICP::pyramid_ICP(bool cuda){

    Eigen::Matrix4f T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[2], *prev_frame_pyramid -> Depth_Pyramid[2], 4, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[1], *prev_frame_pyramid -> Depth_Pyramid[1], 5, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[0], *prev_frame_pyramid -> Depth_Pyramid[0], 10, cuda);
    curr_frame_pyramid -> set_T_gk(T);

    return T;
}

// int main(){

//     FreeImage_Initialise();
//     const char* depth_map_dir_1 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
//     const char* depth_map_dir_2 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.194330.png";
    
//     Frame_Pyramid* frame1 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_1), depth_map_dir_1));
//     frame1->Depth_Pyramid[0]->save_off_format("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/scene1.obj");

//     Frame_Pyramid* frame2 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_2), depth_map_dir_2));
//     frame2->Depth_Pyramid[0]->save_off_format("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/scene2.obj");

//     ICP* icp = new ICP(*frame1, *frame2, 0.1f, 1.1f);
    
//     auto T = icp -> pyramid_ICP(false);

//     std::cout << T;

//     std::vector<Eigen::Vector3f> V_tk;
//     frame1->Depth_Pyramid[0]->apply_transform(T , V_tk);
    
//     std::ofstream OffFile("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/transformed_scene_1.obj");
//     for(auto V : V_tk){
//         OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
//     }
//     OffFile.close();
// }