#include "GPU_icp.h"
#include "../frame/Frame_Pyramid.h"
#include <iostream>
#include <fstream>
#include <FreeImage.h>
#define MINF -std::numeric_limits<float>::infinity()

void ICP::NN_finder(Eigen::Matrix4f source_transformation, Frame & source, const Frame & target, std::vector<Match>& matches){
    // will return the nearest neighbour index for each source vector i.e.
    // we don't take normals into consideration
    // 0 1 2 3 4 5 source vector indices
    // 5 2 3 1 4 4 target vector nearest neighbor to source indices
    // w w w w w w weights of nearest neighbor to source indices
    
    std::vector<Eigen::Vector3f> V_tk; //transformed V

    source.apply_transform(source_transformation, V_tk); //transforming V_k -> V_tk according to source_transformation

    my_NN->buildIndex(target.V_gk); //initializing index for V_gk
    
    matches = my_NN->queryMatches(V_tk);
};

Eigen::Matrix4f ICP::point_to_plane_solver(Frame & source, Frame & target, int iterations, bool cuda){
    
    // source is the live frame F_k and the target is the ray-casted previous frame F_k-1

    Eigen::Matrix4f T_gk_1 = target.T_gk;

    int nPoints = source.M_k1.size();

    bool target_transform_calculated = false;
    
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(nPoints, 6);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(nPoints);

    Eigen::Matrix4f T_gk = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_gk_z;

    std::vector<Eigen::Vector3f> source_vectors = source.V_k; 

    if(!cuda){
        //for loop since first without parallelization
        for(int i = 0; i < nPoints; i++){
            
            target.apply_G_transform();

            std::vector<Match> NN_arr;

            this->NN_finder(T_gk, source, target, NN_arr);
            
            // for(auto i: source.M_k1){
            for(int j = 0; j < source.M_k1.size(); j++){
                //get each point's row in the A matrix     
                
                if(!std::isnan(target.N_gk[NN_arr[j].idx][0]) 
                && !std::isnan(target.N_gk[NN_arr[j].idx][1]) 
                && !std::isnan(target.N_gk[NN_arr[j].idx][2])
                && NN_arr[j].idx != -1){
                    // we have to transform our source.V_k into our estimated transformation matrix
                    Eigen::Vector3f V_gk_source = T_gk.block(0,0,3,3) * source.V_k[source.M_k1[j]] + T_gk.col(3).head(3);
                    
                    // this is innefficient since T_gk_1 and V_k are already known and constant 
                    A(j, 0) = target.N_gk[NN_arr[j].idx][2] * V_gk_source[1] - target.N_gk[NN_arr[j].idx][1] * V_gk_source[2];
                    A(j, 1) = target.N_gk[NN_arr[j].idx][0] * V_gk_source[2] - target.N_gk[NN_arr[j].idx][2] * V_gk_source[0];
                    A(j, 2) = target.N_gk[NN_arr[j].idx][1] * V_gk_source[0] - target.N_gk[NN_arr[j].idx][0] * V_gk_source[1];
                    A(j, 3) = target.N_gk[NN_arr[j].idx][0];
                    A(j, 4) = target.N_gk[NN_arr[j].idx][1];
                    A(j, 5) = target.N_gk[NN_arr[j].idx][2];

                    b(j) = 
                    target.N_gk[NN_arr[j].idx][0] * target.V_gk[NN_arr[j].idx][0] 
                    + target.N_gk[NN_arr[j].idx][1] * target.V_gk[NN_arr[j].idx][1] 
                    + target.N_gk[NN_arr[j].idx][2] * target.V_gk[NN_arr[j].idx][2] 
                    - target.N_gk[NN_arr[j].idx][0] * target.V_gk[NN_arr[j].idx][0]
                    - target.N_gk[NN_arr[j].idx][1] * target.V_gk[NN_arr[j].idx][1]
                    - target.N_gk[NN_arr[j].idx][2] * target.V_gk[NN_arr[j].idx][2];
                }
            }
            target_transform_calculated = true;
            //could be more efficient maybe because ATA is a symmetric matrix
            Eigen::Matrix<float,-1 , -1, Eigen::RowMajor> U = (A.transpose() * A).ldlt().matrixU(); //Upper Triangle of A -> row major according to documentation performance
            Eigen::Vector<float, 6> y = U.triangularView<Eigen::Upper>().solve(A.transpose() * b);
            Eigen::Vector<float, 6> x = U.triangularView<Eigen::Upper>().solve(y);
            

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
    const char* depth_map_dir_2 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.194330.png";
    
    Frame_Pyramid* frame1 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_1), depth_map_dir_1));
    frame1->Depth_Pyramid[0]->save_off_format("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/scene1.obj");

    Frame_Pyramid* frame2 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_2), depth_map_dir_2));
    frame2->Depth_Pyramid[0]->save_off_format("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/scene2.obj");

    ICP* icp = new ICP(*frame1, *frame2, 0.001f);
    
    
    auto T = icp->point_to_plane_solver(*frame1->Depth_Pyramid[0], *frame2->Depth_Pyramid[0], 4, false);

    std::vector<Eigen::Vector3f> V_tk;
    frame1->Depth_Pyramid[0]->apply_transform(T , V_tk);
    
    std::ofstream OffFile("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/transformed_scene_1.obj");
    for(auto V : V_tk){
        OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
    }
    OffFile.close();
}