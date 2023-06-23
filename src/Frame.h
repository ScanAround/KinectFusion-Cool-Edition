#ifndef FRAME_H
#define FRAME_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>



class Frame{

private:

    FIBITMAP * dib;

    //width and height of the image
    int width;
    int height;
    
    //Initial Raw Depth Map
    float * Raw_k;
    //Depth Map
    float * Depth_k; 
    //Calibration Matrix
    Eigen::Matrix3f K_calibration;
    // Vertex Map
    std::vector<Eigen::Vector3f> V_k;
    // Normal Map
    std::vector<Eigen::Vector3f> N_k;
    // Mask Map
    std::vector<int> M_k; // 1 if valid 0 if not valid

public:

    Frame(FIBITMAP & dib);
    
    ~Frame();
    
    Frame(const Frame & from_other);
    
    Frame &operator=(const Frame & Depth_k);
    
    Frame(Frame&& from_other):Depth_k(from_other.Depth_k){};
    
    Frame &operator=(Frame&& from_other);

    FIBITMAP * Apply_Bilateral(const float & sigma_r, const float & float_s, const int & filter_size);

    std::vector<Eigen::Vector3f> calculate_Vks();

    std::vector<Eigen::Vector3f> calculate_Nks();

    void process_image();

    void save_off_format(const std::string & where_to_save);

    inline double N_sigma(const float& sigma, const float &t);
};

#endif