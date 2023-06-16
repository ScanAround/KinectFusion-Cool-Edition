#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H
#include <iostream>
#include <eigen3/Eigen/Dense>

/*

*/

class Pose_Estimator{
private:

    // Vectors V_k and N_k taken from the current frame 

    std::vector<Eigen::Vector3f> V_k;
    std::vector<Eigen::Vector3f> N_k;
    
    // Vectors V_k1 and N_k1 from the previous frame and transformation from previous frame (All taken from raycasting)
    
    std::vector<Eigen::Vector3f> T_gk1;
    std::vector<Eigen::Vector3f> V_k1;
    std::vector<Eigen::Vector3f> N_k1;

    // subsampler method
    // array of float pointers where 0-> points to original image 1-> first subsampling etc. 

    float* Depth_Pyramid[3];
    
    FIBITMAP SubSampler(FIBITMAP * D_k);

public: 
    Pose_Estimator(std::vector<Eigen::Vector3f>& V_k, std::vector<Eigen::Vector3f>& N_k);
    ~Pose_Estimator();
};

#endif