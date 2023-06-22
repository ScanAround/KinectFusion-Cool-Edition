#include "Frame.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>
#include <fstream>
#include <math.h>
#include <array>
#include <vector>


inline double Frame::N_sigma(const float& sigma, const float &t){
    return exp(-std::pow(t,2)*std::pow(sigma,-2));
}

FIBITMAP * Frame::Apply_Bilateral(const float & sigma_r, const float & sigma_s, const int & filter_size){

    Eigen::Matrix3f K_calibration_inverse = K_calibration.inverse();
    FIBITMAP * result = FreeImage_Allocate(640, 480, 16);
    //temporary !!!!! remove this 
    BYTE * result_b = new BYTE[width*height];

    for(int i = static_cast<int>(filter_size/2) ; i < height-(static_cast<int>(filter_size/2)); i++){
        for(int j = static_cast<int>(filter_size/2); j< width-(static_cast<int>(filter_size/2)); j++){
            float sum = 0.0f;
            float normalizing_constant = 0.0f;
            std::cout << Raw_k[i*width + j] << std::endl;
            for(int q = 0; q < filter_size*filter_size; q++){
                sum += N_sigma(sigma_s, sqrt(std::pow(j - (int) q/filter_size, 2) + std::pow(i - q % filter_size, 2))) 
                * N_sigma(sigma_r, Raw_k[i*width + j]-Raw_k[q % filter_size * width + q / filter_size]) 
                * Raw_k[q % filter_size * width + q / filter_size];
                
                std::cout << sqrt(std::pow( -(int) q/filter_size + 1, 2) + std::pow( -(q % filter_size) + 1, 2)) << std::endl;
                
                normalizing_constant += N_sigma(sigma_s, sqrt(std::pow(j - (int) q/filter_size, 2) + std::pow(i - q % filter_size, 2))) 
                * N_sigma(sigma_r, Raw_k[i*width + j]-Raw_k[q % filter_size * width + q / filter_size]) ;
            }
            Depth_k[i*width + j] = sum * 1.0f/normalizing_constant;
            std::cout << Depth_k[i*width + j] << std::endl;
            result_b[i*width + j] = static_cast<uint16_t>(Depth_k[i*width + j]);
            FreeImage_SetPixelIndex(result, j, i, result_b + (i*width + j));
        }
    }

    return result;

}

void Frame::save_off_format(const std::string & where_to_save){

    std::ofstream OffFile(where_to_save + "/vertices.obj");
    for(auto vertix : V_k){
        OffFile << "v " << vertix[0] << " " << vertix[1] << " " << vertix[2] << std::endl; 
    }
    OffFile.close();
}

Frame::Frame(FIBITMAP & dib): dib(FreeImage_ConvertToFloat(&dib)){
    
    width = FreeImage_GetWidth(this->dib);
    height = FreeImage_GetHeight(this->dib);
    
    Raw_k = new float[width*height]; // have to rescale according to the data 
    Depth_k = new float[width*height]; // have to rescale according to the data 
    
    std::memcpy(Raw_k, FreeImage_GetBits(this->dib), width*height);
    
    //dividing by 5000 since scaled by that factor and multiply by 2^16 since stored as 16 bit monochrome images https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
    std::for_each(Raw_k, Raw_k + width*height, [](float x){x * 1.0f/5000.0f *256 * 256;}); 
    
    K_calibration  <<  525.0f, 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;
}

Frame::~Frame(){
    if(dib != nullptr){delete dib;}
    if(Depth_k != nullptr){delete Depth_k;}
    if(Raw_k != nullptr){delete Raw_k;}
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
            
            Eigen::Vector3f ans = (Depth_k[i*width + j])* K_calibration_inverse *  u_dot; 
            V_k.push_back(ans);
            // std::cout << ans[0] << ", " << ans[1] << ", " << ans[2] <<std::endl;
        }
    }

    return V_k;
}

std::vector<Eigen::Vector3f> Frame::calculate_Nks(){
    if(!V_k.empty()){
        for(int i = 0; i < height-1; i++){
            for(int j = 0; j < width-1; j++){
               Eigen::Vector3f ans =(V_k[i*width + j+1] - V_k[(i)*width + j]).cross((V_k[(i+1)*width + j] - V_k[(i)*width + j]));
               ans.normalize();
               N_k.push_back(ans);
            //    std::cout << ans[0] << ", " << ans[1] << ", " << ans[2] <<std::endl;
            }
        }
    }
    else{
        throw std::logic_error("Your V_ks haven't been calculated yet :/");
    }
    return V_k;
}

void Frame::process_image(){
    FIBITMAP * filtered_image = Apply_Bilateral(0.1, 1.0, 3);

    FreeImage_Save(FREE_IMAGE_FORMAT::FIF_UNKNOWN, filtered_image,"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/bilateral_filter.png");
    calculate_Vks();
    calculate_Nks();
}

int main(){
    //sanity check
    FreeImage_Initialise();
    const char * depth_map_dir = "/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031110.534532.png";
    
    Frame * frame1 = new Frame(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir), depth_map_dir));
    
    frame1->process_image();

    frame1->save_off_format("/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition");

}