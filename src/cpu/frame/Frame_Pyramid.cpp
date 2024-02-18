#include "Frame_Pyramid.h"
#include "Frame.h"
#include <iostream>
#include <FreeImage.h>
#include <cmath>

Frame_Pyramid::Frame_Pyramid(FIBITMAP & dib){

    T_gk << Eigen::Matrix4f::Identity();

    Depth_Pyramid[0] = new Frame(dib, T_gk);
    Depth_Pyramid[0] -> process_image();

    Depth_Pyramid[1] = new Frame(*FreeImage_Rescale(Depth_Pyramid[0] -> filtered_dib, 
                                                    Depth_Pyramid[0] -> width/2,
                                                    Depth_Pyramid[0] -> height/2,
                                                    FILTER_BOX), T_gk, 2.0f);
    Depth_Pyramid[1] -> process_image(0.01, 2.0, 10);

    Depth_Pyramid[2] = new Frame(*FreeImage_Rescale(Depth_Pyramid[1] -> filtered_dib, 
                                                    Depth_Pyramid[1] -> width/2,
                                                    Depth_Pyramid[1] -> height/2,
                                                    FILTER_BOX), T_gk, 4.0f);
    Depth_Pyramid[2] -> process_image(0.01, 1.0, 5);
}

Frame_Pyramid::Frame_Pyramid(std::string image_dir_s){

    const char * image_dir = image_dir_s.c_str();

    T_gk << Eigen::Matrix4f::Identity();

    Depth_Pyramid[0] = new Frame(image_dir, T_gk);
    Depth_Pyramid[0] -> process_image();

    Depth_Pyramid[1] = new Frame(*FreeImage_Rescale(Depth_Pyramid[0] -> dib, 
                                                    Depth_Pyramid[0] -> width/2,
                                                    Depth_Pyramid[0] -> height/2,
                                                    FILTER_BOX), T_gk, 2.0f);
    Depth_Pyramid[1] -> process_image(0.01, 2.0, 10);

    Depth_Pyramid[2] = new Frame(*FreeImage_Rescale(Depth_Pyramid[1] -> dib, 
                                                    Depth_Pyramid[1] -> width/2,
                                                    Depth_Pyramid[1] -> height/2,
                                                    FILTER_BOX), T_gk, 4.0f);
    Depth_Pyramid[2] -> process_image(0.01, 1.0, 5);
}

Frame_Pyramid::Frame_Pyramid(std::vector<Eigen::Vector3f> V_gks, std::vector<Eigen::Vector3f> N_gks, Eigen::Matrix4f T_gk, int width, int height):
T_gk(T_gk){

    Depth_Pyramid[0] = new Frame(V_gks, N_gks, T_gk, width, height);
    Depth_Pyramid[1] = new Frame(V_gks, N_gks, T_gk, width, height);
    Depth_Pyramid[2] = new Frame(V_gks, N_gks, T_gk, width, height);
    this -> set_T_gk(T_gk);

}

Frame_Pyramid::Frame_Pyramid(std::vector<float> depthMap,
                             Eigen::Matrix3f K, Eigen::Matrix4f T_gk,
                             int width, int height){
        
    Depth_Pyramid[0] = new Frame(depthMap, T_gk, K, width, height, 1.0f);
    Depth_Pyramid[0] -> calculate_Vks();
    Depth_Pyramid[0] -> calculate_Nks();

    Depth_Pyramid[1] = new Frame(depthMap, T_gk, K, width, height, 2.0f);
    Depth_Pyramid[1] -> calculate_Vks();
    Depth_Pyramid[1] -> calculate_Nks();

    Depth_Pyramid[2] = new Frame(depthMap, T_gk, K, width, height, 4.0f);
    Depth_Pyramid[2] -> calculate_Vks();
    Depth_Pyramid[2] -> calculate_Nks();

    this -> set_T_gk(T_gk);
}


Frame_Pyramid::~Frame_Pyramid(){
    delete Depth_Pyramid[0];
    Depth_Pyramid[0] = nullptr;
    delete Depth_Pyramid[1];
    Depth_Pyramid[1] = nullptr;
    delete Depth_Pyramid[2];
    Depth_Pyramid[2] = nullptr;

}