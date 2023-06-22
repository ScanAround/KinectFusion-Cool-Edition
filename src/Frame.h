#ifndef FRAME_H
#define FRAME_H
#include "Pose_Estimation.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>

class Frame{

private:
    //Depth Map
    const FIBITMAP Depth_k;
    //Calibration Matrix
    const Eigen::Matrix3f K_calibration; 
    // Vertex Map
    const std::vector<Eigen::Vector3f> V_k;
    // Normal Map
    const std::vector<Eigen::Vector3f> N_k;
    // Mask Map
    const std::vector<Eigen::Vector3f> M_k;

public:

    Frame(const FIBITMAP & Depth_k): Depth_k(Depth_k){};
    
    ~Frame();
    
    Frame(const Frame & from_other):Depth_k(from_other.Depth_k){};
    
    Frame &operator=(const Frame & Depth_k);
    
    Frame(Frame&& from_other):Depth_k(from_other.Depth_k){};
    
    Frame &operator=(Frame&& from_other);

    std::array<std::vector<Eigen::Vector3f>,3> get_Vs_Ns_Ms();

    FIBITMAP Apply_Bilateral(const int & paramaters);

    Eigen::Vector3f* calculate_Vks();

    Eigen::Vector3f* calculate_Nks();

    Eigen::Vector3f* calculate_Mks();
};

#endif