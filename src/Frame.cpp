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
    // //temporary!!!!!!!
    // //remove this
    FIBITMAP * result;
    // result = FreeImage_Allocate(width, height, 8); // monochrome image therefore 8 bytes
    // BYTE * image_data = FreeImage_GetBits(result);


    for(int i = static_cast<int>(filter_size/2) ; i < height-(static_cast<int>(filter_size/2)); ++i){
        for(int j = static_cast<int>(filter_size/2); j < width-(static_cast<int>(filter_size/2)); ++j){
            float sum = 0.0f;
            float normalizing_constant = 0.0f;
            for(int q = 0; q < filter_size*filter_size; ++q){
                sum += N_sigma(sigma_s,  sqrt(std::pow( -(int) q/filter_size + static_cast<int>(filter_size/2), 2) + std::pow( -(q % filter_size) + static_cast<int>(filter_size/2), 2)))
                * N_sigma(sigma_r, Raw_k[i*width + j]-Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)])
                * Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)];
      
                normalizing_constant += N_sigma(sigma_s,  sqrt(std::pow( -(int) q/filter_size + static_cast<int>(filter_size/2), 2) + std::pow( -(q % filter_size) + static_cast<int>(filter_size/2), 2)))
                * N_sigma(sigma_r, Raw_k[i*width + j]-Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)]);
            }
            
            Depth_k[i*width + j] = sum/normalizing_constant;
            // image_data[i*width + j] = static_cast<BYTE>(Depth_k[i*width + j]*255.0f);  // done to see filtered image
        }
    }
    return result;

}

void Frame::save_off_format(const std::string & where_to_save){

    std::ofstream OffFile(where_to_save + "/vertices.obj");
    for(int i = 0; i < V_k.size();i++){
        OffFile << "v " << V_k[i][0] << " " << V_k[i][1] << " " << V_k[i][2] << std::endl; 
        OffFile << "vn " << N_k[i][0] << " " << N_k[i][1] << " " << N_k[i][2] << std::endl; 
    }
    OffFile.close();
}

Frame::Frame(FIBITMAP & dib): dib(FreeImage_ConvertToFloat(&dib)){
    
    width = FreeImage_GetWidth(this->dib);
    height = FreeImage_GetHeight(this->dib);
    
    Depth_k = new float[width*height]; // have to rescale according to the data 

    Raw_k = (float *) FreeImage_GetBits(this->dib) ; // have to rescale according to the data 
    
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
            
            //dividing by 5000 since scaled by that factor https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
            Eigen::Vector3f ans = (Depth_k[i*width + j]/ 5000.0f *255.0f * 255.0f)* K_calibration_inverse *  u_dot; 
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
    Apply_Bilateral(0.01, 3.0, 15);

    // FreeImage_Save(FREE_IMAGE_FORMAT::FIF_PNG, filtered_image,"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/bilateral_filter.png");
    calculate_Vks();
    calculate_Nks();
}

int main(){
    //sanity check
    FreeImage_Initialise();
    const char * depth_map_dir = "/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    
    Frame * frame1 = new Frame(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir), depth_map_dir));
    
    frame1->process_image();

    // frame1->save_off_format("/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition");

}