#ifndef FRAME_H
#define FRAME_H
#include "Pose_Estimation.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>



class Frame{

private:

    FIBITMAP * dib;

    //width and height of the image
    int width;
    int height;
    
    //Depth Map
    float * Depth_k;
    //Calibration Matrix
    Eigen::Matrix3f K_calibration;
    // Vertex Map
    std::vector<Eigen::Vector3f> V_k;
    // Normal Map
    std::vector<Eigen::Vector3f> N_k;
    // Mask Map
    std::vector<Eigen::Vector3f> M_k;

public:

    Frame(FIBITMAP & dib);
    
    ~Frame();
    
    Frame(const Frame & from_other);
    
    Frame &operator=(const Frame & Depth_k);
    
    Frame(Frame&& from_other):Depth_k(from_other.Depth_k){};
    
    Frame &operator=(Frame&& from_other);

    FIBITMAP Apply_Bilateral(const int & paramaters);

    std::vector<Eigen::Vector3f> calculate_Vks();

    std::vector<Eigen::Vector3f> calculate_Nks();

    void process_image();

    void save_off_format(const std::string & where_to_save);
};

#endif