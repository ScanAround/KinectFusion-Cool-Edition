#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h> 

/*

*/

class Pose_Estimator{
private:

    const FIBITMAP * D_k;

    // Vectors V_k and N_k taken from the current frame 

    const std::vector<Eigen::Vector3f> * V_k;
    const std::vector<Eigen::Vector3f> * N_k;
    
    // Vectors V_k1 and N_k1 from the previous frame and transformation from previous frame (All taken from raycasting)
    
    const std::vector<Eigen::Vector3f> * T_gk1;
    const std::vector<Eigen::Vector3f> * V_k1;
    const std::vector<Eigen::Vector3f> * N_k1;

    // subsampler method
    // array of float pointers where 0-> points to original image 1-> first subsampling etc. 

   

    FIBITMAP* Depth_Pyramid[3];
    
    FIBITMAP * SubSampler(FIBITMAP * D_k);


public: 

    FIBITMAP get_D_k() const;
    FIBITMAP* Pyramid_Generator();
    Pose_Estimator(std::vector<Eigen::Vector3f> * V_k, std::vector<Eigen::Vector3f> * N_k, FIBITMAP * D_k): V_k(V_k), N_k(N_k), D_k(D_k){};
    ~Pose_Estimator();
};

#endif