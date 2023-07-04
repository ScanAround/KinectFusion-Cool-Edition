#ifndef FRAME_H
#define FRAME_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>

class Frame{

private:

    FIBITMAP * dib;
    
    
    //Initial Raw Depth Map
    float * Raw_k;
    //Depth Map
    float * Depth_k; 
    //Calibration Matrix

public:
    Eigen::Vector4f T_gk; // transformation matrix to global coordinates of the current frame
    
    // Vertex Map
    std::vector<Eigen::Vector3f> V_k;
    // Normal Map
    std::vector<Eigen::Vector3f> N_k;
    // Mask Map
    std::vector<int> M_k; // 1 if valid 0 if not valid 
    // probably should make a map instead of a std::vector here
    
    Eigen::Matrix3f K_calibration;
    
    FIBITMAP * filtered_dib;

    //width and height of the image
    int width;
    int height;


    Frame(FIBITMAP & dib, float sub_sampling_rate = 1.0f);
    
    ~Frame();
    
    Frame(const Frame & from_other);
    
    Frame &operator=(const Frame & Depth_k);
    
    Frame(Frame&& from_other):Depth_k(from_other.Depth_k){};
    
    Frame &operator=(Frame&& from_other);

    FIBITMAP * Apply_Bilateral(const float & sigma_r, const float & float_s, const int & filter_size);

    std::vector<Eigen::Vector3f> calculate_Vks();

    std::vector<Eigen::Vector3f> calculate_Nks();

    void process_image(float sigma_r = 0.01, float sigma_s = 3.0,  int filter_size = 15, bool apply_bilateral = true);

    void save_off_format(const std::string & where_to_save);

    inline double N_sigma(const float& sigma, const float &t);
};

#endif