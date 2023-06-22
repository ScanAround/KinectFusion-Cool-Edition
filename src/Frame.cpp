#include "Frame.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>

Frame::Frame(FIBITMAP & dib): dib(&dib){
    
    width = FreeImage_GetWidth(this->dib);
    height = FreeImage_GetHeight(this->dib);
    
    Depth_k = (float *)FreeImage_GetBits(this->dib);
}

Frame::~Frame(){
    if(dib != nullptr){delete dib;}
    if(Depth_k != nullptr){delete Depth_k;}
}

Frame::Frame(const Frame & from_other): Depth_k(from_other.Depth_k){
    if(!from_other.V_k.empty() and !from_other.M_k.empty() and !from_other.N_k.empty()){
        V_k = from_other.V_k;
        N_k = from_other.N_k;
        M_k = from_other.M_k;
    }
    else{
        throw std::logic_error("Either V_k vector, N_k vector, or M_k vectors are not initialized");
    }
}

std::vector<Eigen::Vector3f> Frame::calculate_Vks(){
    
    Eigen::Matrix3f K_calibration_inverse = K_calibration.inverse();
    
    Eigen::Vector3f u_dot;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            u_dot << j, i ,1;
            V_k.push_back(Depth_k[i*width + j] * K_calibration_inverse *  u_dot);
        }
    }

    return V_k;
}