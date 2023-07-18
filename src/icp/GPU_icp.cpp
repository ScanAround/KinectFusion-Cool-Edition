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

    curr_frame.apply_transform(T_curr_frame, curr_V_k_transformed); //transforming current frame's V_k -> V_tk according to previous frame

    std::cout << "Finding Correspondences";

    for(int i = 0; i < curr_V_k_transformed.size(); i++){
        if(!std::isnan(curr_V_k_transformed[i][0])){
            Eigen::Vector2i pixel = prev_frame.vec_to_pixel(curr_V_k_transformed[i]);
            int idx_in_prev = pixel[1]*prev_frame.width + pixel[0];
            
            //check if idx_in_prev is a valid vertex in previous frame
            if(!std::isnan(prev_frame.V_gk[idx_in_prev][0])){
                matches.push_back(std::make_pair(i, idx_in_prev));
            }
        }
    }
}

Eigen::Matrix4f ICP::point_to_plane_solver(Frame & curr_frame, Frame & prev_frame, int iterations, bool cuda){
    
    // source is the live frame F_k and the prev_frame is the ray-casted previous frame F_k-1

    Eigen::Matrix4f T_gk_1 = prev_frame.T_gk;

    bool prev_frame_transform_calculated = false;

    Eigen::Matrix4f T_gk = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_gk_z;

    std::vector<Eigen::Vector3f> source_vectors = curr_frame.V_k; 

    if(!cuda){
        //for loop since first without parallelization
        for(int i = 0; i < iterations; i++){
            
            prev_frame.apply_G_transform();

            std::vector<std::pair<int, int>> correspondences;

            this->correspondence_finder(T_gk, curr_frame, prev_frame, correspondences);
            
            int corr_size = correspondences.size();
            
            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(corr_size, 6);
            Eigen::VectorXf b = Eigen::VectorXf::Zero(corr_size);
            
            // for(auto i: source.M_k1){
            for(int j = 0; j < corr_size; j++){
                // get each point's row in the A matrix     
                // we have to transform our source.V_k into our estimated transformation matrix
                Eigen::Vector3f s_i = T_gk.block(0,0,3,3) * curr_frame.V_k[correspondences[j].first] + T_gk.block(0,3,3,1);
                Eigen::Vector3f d_i = prev_frame.V_gk[correspondences[j].second];
                
                Eigen::Vector3f cross_i = s_i.cross(d_i);

                Eigen::Vector3f n_i = prev_frame.N_gk[correspondences[j].second];
                
                // this is innefficient since T_gk_1 and V_k are already known and constant 
                A(j, 0) = cross_i[0];
                A(j, 1) = cross_i[1];
                A(j, 2) = cross_i[2];
                A(j, 3) = n_i[0];
                A(j, 4) = n_i[1];
                A(j, 5) = n_i[2];

                b(j) = n_i.dot(d_i) - n_i.dot(s_i);
                
            }
            //could be more efficient maybe because ATA is a symmetric matrix

            Eigen::Vector<float, 6> x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

            // Eigen::Matrix<float,-1 , -1, Eigen::RowMajor> U = (A.transpose() * A).ldlt().matrixU(); //Upper Triangle of A -> row major according to documentation performance
            // Eigen::Vector<float, 6> y = U.triangularView<Eigen::Upper>().solve(A.transpose() * b);
            // Eigen::Vector<float, 6> x = U.triangularView<Eigen::Upper>().solve(y);
            

            T_gk_z <<      1 ,  x[2] , -x[1] , x[3],
                       -x[2] ,   1   ,  x[0] , x[4],
                        x[1] , -x[0] ,   1   , x[5],
                        0    ,  0    ,   0   ,  1  ; 
            
            T_gk = T_gk_z * T_gk;
            
        }
        
    }

    else{
        //parallelization part
    }
    
    return T_gk;
    
    }

int main(){

    FreeImage_Initialise();
    const char* depth_map_dir_1 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    const char* depth_map_dir_2 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.295279.png";
    
    Frame_Pyramid* frame1 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_1), depth_map_dir_1));
    frame1->Depth_Pyramid[0]->save_off_format("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/scene1.obj");

    Frame_Pyramid* frame2 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_2), depth_map_dir_2));
    frame2->Depth_Pyramid[0]->save_off_format("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/scene2.obj");

    ICP* icp = new ICP(*frame1, *frame2, 0.001f);
    
    
    auto T = icp->point_to_plane_solver(*frame1->Depth_Pyramid[0], *frame2->Depth_Pyramid[0], 4, false);
    
    std::cout << T;

    std::vector<Eigen::Vector3f> V_tk;
    frame1->Depth_Pyramid[0]->apply_transform(T , V_tk);
    
    std::ofstream OffFile("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/transformed_scene_1.obj");
    for(auto V : V_tk){
        OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
    }
    OffFile.close();
}