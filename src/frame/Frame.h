#ifndef FRAME_H
#define FRAME_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>
#include <map>

class Frame{


public:

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

    // takes a vector in the global frame and backprojects it to camera pixels
    Eigen::Vector2i vec_to_pixel(const Eigen::Vector3f& vec){
        Eigen::Matrix3f rotation = T_gk.inverse().block(0,0,3,3);
        Eigen::Vector3f translation = T_gk.inverse().block(0,3,3,1);
        
        Eigen::Vector3f vec_camera_frame = rotation * vec + translation;
        
        Eigen::Vector3f u_dot = (K_calibration * vec_camera_frame) / vec_camera_frame[2];

        Eigen::Vector2i u;
        u << int(u_dot[0]), int(u_dot[1]);
        
        return u;
    };

    void apply_G_transform(){
        if(!transformed){
            for(auto idx:M_k1){
                V_gk.push_back(T_gk.block(0,0,3,3) * V_k[idx] + T_gk.col(3).head(3)); 
                N_gk.push_back(T_gk.block(0,0,3,3) * N_k[idx] + T_gk.col(3).head(3)); 
            }
        }
        transformed = true;
    };

    void apply_transform(Eigen::Matrix4f T, std::vector<Eigen::Vector3f>& V_tk){
        for(auto idx:M_k1){
            V_tk.push_back(T.block(0,0,3,3) * V_k[idx] + T.block(0,3,3,1)); 
        }
    };

    inline double N_sigma(const float& sigma, const float &t);

    // transformation matrix to global coordinates of the current frame
    Eigen::Matrix4f T_gk; 
    
    // Vertex Map
    std::vector<Eigen::Vector3f> V_k;
    // Normal Map
    std::vector<Eigen::Vector3f> N_k;

    // Coordinate Transform of Vertex Map to global coordinate (only calculated if called)
    std::vector<Eigen::Vector3f> V_gk;
    // Coordinate Transform of Normal Map to global coordinate (only calculated if called)
    std::vector<Eigen::Vector3f> N_gk;
    
    // Used to know if apply_transform was already applied
    bool transformed = false;

    // Mask Map
    std::vector<int> M_k0; 
    std::vector<int> M_k1; 

    // M_k1 encodes valid vector indices
    // M_k0 encodes invalid vector indices
    
    Eigen::Matrix3f K_calibration;
    
    FIBITMAP * filtered_dib;

    //width and height of the image
    int width;
    int height;

private:

    FIBITMAP * dib;
    
    //Initial Raw Depth Map
    float * Raw_k;
    //Depth Map
    float * Depth_k; 
    //Calibration Matrix

};

#endif