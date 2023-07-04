#include "GPU_icp.h"

//initially without GPU implementation 

Eigen::Vector4f ICP::point_to_plane_solver(const Frame & source, const Frame & target, int iterations, bool cuda, Eigen::Vector4f T_gk_1){
    
    // source is the live frame F_k and the target is the ray-casted previous frame F_k-1

    
    int nPoints = source.M_k.size();

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(nPoints, 6);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(nPoints);

    Eigen::Matrix4f T_gk = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_gk_z;

    std::vector<Eigen::Vector3f> source_vectors = source.V_k; 
    std::vector<Eigen::Vector3f> source_vectors = source.V_k; 

    if(!cuda){
        //for loop since first without parallelization
        for(int i = 0; i < iterations; i++){
            
            std::unique_ptr<int> NN_correspondences = this->NN_finder(source_vectors, target.V_k); //YOU HAVE TO TRANSFORM SOURCE AND TARGET TO G COORDINATE SYSTEM!!!!!!
            
            for(int i = 0; i < nPoints; i++){
                //get each point's row in the A matrix     
                


                // we have to transform our source.V_k into our estimated transformation matrix
                Eigen::Vector3f V_gk_source = T_gk.block(0,0,3,3) * source.V_k[i] + T_gk.col(3).head(3);
                
                Eigen::Vector3f V_gk_target = T_gk_1.block(0,0,3,3) * target.V_k[i] + T_gk_1.col(3).head(3);
                Eigen::Vector3f N_gk_target = T_gk_1.block(0,0,3,3) * target.N_k[i] + T_gk_1.col(3).head(3);
                
                // we also should only go through valid V_ks!!!!

                A(i, 0) = N_gk_target[2] * V_gk_source[1] - N_gk_target[1] * V_gk_source[2];
                A(i, 1) = N_gk_target[0] * V_gk_source[2] - N_gk_target[2] * V_gk_source[0];
                A(i, 2) = N_gk_target[1] * V_gk_source[0] - N_gk_target[0] * V_gk_source[1];
                A(i, 3) = N_gk_target[0];
                A(i, 4) = N_gk_target[1];
                A(i, 5) = N_gk_target[2];

                b(i) = 
                  N_gk_target[0] * V_gk_target[0] 
                + N_gk_target[1] * V_gk_target[1] 
                + N_gk_target[2] * V_gk_target[2] 
                - N_gk_target[0] * V_gk_source[0]
                - N_gk_target[1] * V_gk_source[1]
                - N_gk_target[2] * V_gk_source[2];
            }

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

            Eigen::Matrix<float,-1 , -1, Eigen::RowMajor> U = (A.transpose() * A).ldlt().matrixU(); //Upper Triangle of A -> row major according to documentation performance
            Eigen::Vector<float, 6> y = U.triangularView<Eigen::Upper>().solve(A.transpose() * b);
            Eigen::Vector<float, 6> x = U.triangularView<Eigen::Upper>().solve(y);
    }
    }