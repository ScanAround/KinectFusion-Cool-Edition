#include "GPU_icp.h"

//initially without GPU implementation 

Eigen::Vector4f ICP::point_to_plane_solver(const Frame & source, const Frame & target, int iterations){
    
    // source is the live frame F_k and the target is the ray-casted previous frame F_k-1

    std::unique_ptr<int> NN_correspondences = this->NN_finder(source, target);
    
    int nPoints = source.M_k.size();

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(nPoints, 6);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(nPoints);


    //for loop since first without parallelization

    for(int i = 0; i < nPoints; i++){
        
        A(i, 0) = target.N_k[i][2] * source.V_k[i][1] - target.N_k[i][1] * source.V_k[i][2];
        A(i, 1) = target.N_k[i][0] * source.V_k[i][2] - target.N_k[i][2] * source.V_k[i][0];
        A(i, 2) = target.N_k[i][1] * source.V_k[i][0] - target.N_k[i][0] * source.V_k[i][1];
        A(i, 3) = target.N_k[i][0];
        A(i, 4) = target.N_k[i][1];
        A(i, 5) = target.N_k[i][2];

        b(i) = target.N_k[i][0] * target.V_k[i][0] 
        + target.N_k[i][1] * target.V_k[i][1] 
        + target.N_k[i][2] * target.V_k[i][2] 
        - target.N_k[i][0] * source.V_k[i][0]
        - target.N_k[i][1] * source.V_k[i][1]
        - target.N_k[i][2] * source.V_k[i][2];

    }

}