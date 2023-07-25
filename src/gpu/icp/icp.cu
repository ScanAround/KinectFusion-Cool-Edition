#include "../../cpu/icp/ICP.h"
#include "../../cpu/frame/Frame_Pyramid.h"
#include <iostream>
#include <fstream>
#include <FreeImage.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#ifndef CUDACC
#define CUDACC
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#define BLOCKSIZE_REDUCED 256
#define MINF -std::numeric_limits<float>::infinity()
using namespace Eigen;

void correspondence_finder_cuda_cpu(Vector3f* verticesSource, Vector3f* verticesPrevious, Vector3f* normalsSource, Vector3f* normalsPrevious,
    Matrix4f estimatedPose, Matrix4f transMatrixcur, Matrix3f K_calibration,
    int width, int height, float distance_threshold, float angle_threshold, Eigen::MatrixXf &A, Eigen::MatrixXf& b) {
    // will return the nearest neighbour index for each source vector i.e.
    // we don't take normals into consideration
    // 0 1 2 3 4 5 source vector indices
    // 5 2 3 1 4 4 prev_frame vector nearest neighbor to source indices
   // int tid = (blockIdx.x * blockDim.x) + threadIdx.x;


    for (int i = 0; i < width*height; i++) {
        Vector3f curr_V_k_transformed = transMatrixcur.block(0, 0, 3, 3) * verticesSource[i] + transMatrixcur.block(0, 3, 3, 1); //transforming current frame's V_k -> V_tk according to previous frame
        Vector3f curr_N_k_transformed = transMatrixcur.block(0, 0, 3, 3) * normalsSource[i]; //transforming current frame's V_k -> V_tk according to previous frame

        if (!isnan(curr_V_k_transformed[0])) {
            Eigen::Matrix3f rotation = estimatedPose.inverse().block(0, 0, 3, 3);
            Eigen::Vector3f translation = estimatedPose.inverse().block(0, 3, 3, 1);

            Eigen::Vector3f vec_camera_frame = rotation * curr_V_k_transformed + translation;

            Eigen::Vector3f u_dot = (K_calibration * vec_camera_frame) / vec_camera_frame[2];

            Eigen::Vector2i pixel;
            if (u_dot[0] >= 0
                && u_dot[0] <= width
                && u_dot[1] >= 0
                && u_dot[1] <= height) {
                // making sure u is within the image we have 
                pixel << int(u_dot[0]), int(u_dot[1]);
            }
            else {
                pixel << 0, 0;
            }
            int idx_in_prev = pixel[1] * width + pixel[0];
            Vector3f prev_V_k_transformed = estimatedPose.block(0, 0, 3, 3) * verticesPrevious[idx_in_prev] + estimatedPose.block(0, 3, 3, 1); //transforming current frame's V_k -> V_tk according to previous frame
            Vector3f prev_N_k_transformed = estimatedPose.block(0, 0, 3, 3) * normalsPrevious[idx_in_prev]; //transforming current frame's V_k -> V_tk according to previous frame

            //Normals aren't calculated in the last row => check if idx is less than last row idxs
            if (idx_in_prev < (height - 1) * width) {
                //check if idx_in_prev is a valid vertex in previous frame
                if (!isnan(prev_N_k_transformed[0])) {
                    // check if distances between corresponding vectors are below threshold
                    if ((curr_V_k_transformed - prev_V_k_transformed).norm() <= distance_threshold) {
                        // check if distances between corresponding angles are below threshold
                        if ((curr_N_k_transformed.dot(prev_N_k_transformed)) <= angle_threshold) {
                            // get each point's contribution to ATA
                            // we have to transform our source.V_k into our estimated transformation matrix
                            Eigen::Vector3f s_i = curr_V_k_transformed;
                            Eigen::Vector3f d_i = prev_V_k_transformed;

                            Eigen::Vector3f n_i = prev_N_k_transformed;

                            Eigen::MatrixXf A_jT = Eigen::MatrixXf::Zero(6, 1);
                            A_jT.block(0, 0, 3, 1) = s_i.cross(n_i);
                            A_jT.block(3, 0, 3, 1) = n_i;

                            A.selfadjointView<Eigen::Lower>().rankUpdate(A_jT); // only calculates the lower triangle (eigen's ldlt calculates with lower triangle) and adds

                            b += A_jT * (n_i.dot(d_i) - n_i.dot(s_i));
                        }
                    }
                }
            }
        }
    }
}
__global__ void correspondence_finder_cuda(int* count, Vector3f* verticesSource, Vector3f* verticesPrevious, Vector3f* normalsSource, Vector3f* normalsPrevious,
    Matrix4f estimatedPose, Matrix4f transMatrixcur, Matrix3f K_calibration,
    int width,int height,float distance_threshold, float angle_threshold , Eigen::MatrixXf A, Eigen::MatrixXf b) {
    // will return the nearest neighbour index for each source vector i.e.
    // we don't take normals into consideration
    // 0 1 2 3 4 5 source vector indices
    // 5 2 3 1 4 4 prev_frame vector nearest neighbor to source indices
   // int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    *count = 0;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < width*height) {
        int i = tid;
        Vector3f curr_V_k_transformed = transMatrixcur.block(0, 0, 3, 3) * verticesSource[i] + transMatrixcur.block(0, 3, 3, 1); //transforming current frame's V_k -> V_tk according to previous frame
        Vector3f curr_N_k_transformed = transMatrixcur.block(0, 0, 3, 3) * normalsSource[i]; //transforming current frame's V_k -> V_tk according to previous frame

        if (!isnan(curr_V_k_transformed[0])) {
            Eigen::Matrix3f rotation = estimatedPose.inverse().block(0, 0, 3, 3);
            Eigen::Vector3f translation = estimatedPose.inverse().block(0, 3, 3, 1);

            Eigen::Vector3f vec_camera_frame = rotation * curr_V_k_transformed + translation;

            Eigen::Vector3f u_dot = (K_calibration * vec_camera_frame) / vec_camera_frame[2];

            Eigen::Vector2i pixel;
            if (u_dot[0] >= 0
                && u_dot[0] <= width
                && u_dot[1] >= 0
                && u_dot[1] <= height) {
                // making sure u is within the image we have 
                pixel << int(u_dot[0]), int(u_dot[1]);
            }
            else {
                pixel << 0, 0;
            }
            int idx_in_prev = pixel[1] * width + pixel[0];
            Vector3f prev_V_k_transformed = estimatedPose.block(0, 0, 3, 3) * verticesPrevious[idx_in_prev] + estimatedPose.block(0, 3, 3, 1); //transforming current frame's V_k -> V_tk according to previous frame
            Vector3f prev_N_k_transformed = estimatedPose.block(0, 0, 3, 3) * normalsPrevious[idx_in_prev]; //transforming current frame's V_k -> V_tk according to previous frame

            //Normals aren't calculated in the last row => check if idx is less than last row idxs
            if (idx_in_prev < (height - 1) * width) {
                //check if idx_in_prev is a valid vertex in previous frame
                if (!isnan(prev_N_k_transformed[0])) {
                    // check if distances between corresponding vectors are below threshold
                    if ((curr_V_k_transformed - prev_V_k_transformed).norm() <= distance_threshold) {
                        // check if distances between corresponding angles are below threshold
                        if ((curr_N_k_transformed.dot(prev_N_k_transformed)) <= angle_threshold) {
                            // get each point's contribution to ATA
                            // we have to transform our source.V_k into our estimated transformation matrix
                            Eigen::Vector3f s_i = curr_V_k_transformed;
                            Eigen::Vector3f d_i = prev_V_k_transformed;

                            Eigen::Vector3f n_i = prev_N_k_transformed;

                            Eigen::MatrixXf A_jT = Eigen::MatrixXf::Zero(6, 1);
                            A_jT.block(0, 0, 3, 1) = s_i.cross(n_i);
                            A_jT.block(3, 0, 3, 1) = n_i;

                            A.selfadjointView<Eigen::Lower>().rankUpdate(A_jT); // only calculates the lower triangle (eigen's ldlt calculates with lower triangle) and adds

                            b += A_jT * (n_i.dot(d_i) - n_i.dot(s_i));
                        }
                    }
                }
            }
        }
    }
}

void ICP::correspondence_finder(Eigen::Matrix4f T_curr_frame, Frame & curr_frame, Frame & prev_frame, std::vector<std::pair<int, int>>& matches){
    // will return the nearest neighbour index for each source vector i.e.
    // we don't take normals into consideration
    // 0 1 2 3 4 5 source vector indices
    // 5 2 3 1 4 4 prev_frame vector nearest neighbor to source indices
    
    std::vector<Eigen::Vector3f> curr_V_k_transformed; //transformed V
    std::vector<Eigen::Vector3f> curr_N_k_transformed; //transformed V

    curr_frame.apply_transform(T_curr_frame, curr_V_k_transformed, curr_N_k_transformed); //transforming current frame's V_k -> V_tk according to previous frame

    std::cout << "Finding Correspondences" << std::endl;

    for(int i = 0; i < curr_V_k_transformed.size(); i++){
        if(!std::isnan(curr_V_k_transformed[i][0])){
            Eigen::Vector2i pixel = prev_frame.vec_to_pixel(curr_V_k_transformed[i]);
            int idx_in_prev = pixel[1]*prev_frame.width + pixel[0];
            
            //Normals aren't calculated in the last row => check if idx is less than last row idxs
            if(idx_in_prev < (prev_frame.height-1) * prev_frame.width){
                //check if idx_in_prev is a valid vertex in previous frame
                if(!std::isnan(prev_frame.N_gk[idx_in_prev][0])){
                    // check if distances between corresponding vectors are below threshold
                    if((curr_V_k_transformed[i] - prev_frame.V_gk[idx_in_prev]).norm() <= this->distance_threshold){
                        // check if distances between corresponding angles are below threshold
                        if((curr_N_k_transformed[i].dot(prev_frame.N_gk[idx_in_prev])) <= this->angle_threshold){
                            matches.push_back(std::make_pair(i, idx_in_prev));
                            std::cout << i << " "<< idx_in_prev<< "\n";
                        }
                    }
                }
            }
        }
    }
}


Eigen::Matrix4f ICP::point_to_plane_solver(Frame & curr_frame, Frame & prev_frame, int iterations, bool cuda){
    
    // source is the live frame F_k and the prev_frame is the ray-casted previous frame F_k-1
    Eigen::Matrix4f T_gk_z = curr_frame.T_gk;

    std::vector<Eigen::Vector3f> source_vectors = curr_frame.V_k; 

    if(!cuda){
        int width = curr_frame.width;
        int height = curr_frame.height;


        Eigen::MatrixXf A_final = Matrix<float, 6, 6>::Zero();

        Eigen::MatrixXf b_final = Matrix<float, 6, 1>::Zero();

        // cudaMemcpy(&K_calibration, prev_frame.K_calibration.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice);
        Matrix4f estimatedPose = prev_frame.T_gk;
        Matrix4f transMatrixcur = curr_frame.T_gk;
        Matrix3f K_calibration = prev_frame.K_calibration;
        float distance_threshold = 0.1f;
        float angle_threshold = 1.1f;
        int* count = 0;
        int* count_cpu = 0;
        
        for (int i = 0; i < iterations; i++)
        {
            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6); // supposed to be ATA of the system according to paper


            Eigen::MatrixXf b = Eigen::MatrixXf::Zero(6, 1);

            correspondence_finder_cuda_cpu(curr_frame.V_k.data(), prev_frame.V_k.data(), curr_frame.N_k.data(), prev_frame.N_k.data(),
                estimatedPose, transMatrixcur, K_calibration,
                width, height, distance_threshold, angle_threshold, A, b);

            Eigen::Vector<float, 6> x = A.ldlt().solve(b); //ldlt because ATA not always Positive Definite

            float alpha = x[0];
            float beta = x[1];
            float gamma = x[2];


            T_gk_z << 1, alpha* beta - gamma, alpha* gamma + beta, x[3],
                gamma, alpha* beta* gamma + 1, beta* gamma - alpha, x[4],
                -beta, alpha, 1, x[5],
                0, 0, 0, 1;
        }
        
        /*
        for (int i = 0; i < iterations; i++) {

            prev_frame.apply_G_transform();

            std::vector<std::pair<int, int>> correspondences;

            this->correspondence_finder(T_gk_z, curr_frame, prev_frame, correspondences);

            int corr_size = correspondences.size();

            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6); // supposed to be ATA of the system according to paper

            Eigen::MatrixXf A_jT = Eigen::MatrixXf::Zero(6, 1);

            Eigen::MatrixXf b = Eigen::MatrixXf::Zero(6, 1); // supposed to be ATb according to paper
           
            // for(auto i: source.M_k1){
            for (int j = 0; j < corr_size; ++j) {
                // get each point's contribution to ATA
                // we have to transform our source.V_k into our estimated transformation matrix
                Eigen::Vector3f s_i = T_gk_z.block(0, 0, 3, 3) * curr_frame.V_k[correspondences[j].first] + T_gk_z.block(0, 3, 3, 1);
                Eigen::Vector3f d_i = prev_frame.V_gk[correspondences[j].second];

                Eigen::Vector3f n_i = prev_frame.N_gk[correspondences[j].second];

                A_jT.block(0, 0, 3, 1) = s_i.cross(n_i);
                A_jT.block(3, 0, 3, 1) = n_i;

                A.selfadjointView<Eigen::Lower>().rankUpdate(A_jT); // only calculates the lower triangle (eigen's ldlt calculates with lower triangle) and adds

                b += A_jT * (n_i.dot(d_i) - n_i.dot(s_i));

            }
            // std::cout << A << std::endl;
            // std::cout << b << std::endl;

            Eigen::Vector<float, 6> x = A.ldlt().solve(b); //ldlt because ATA not always Positive Definite

            float alpha = x[0];
            float beta = x[1];
            float gamma = x[2];


            T_gk_z << 1, alpha* beta - gamma, alpha* gamma + beta, x[3],
                gamma, alpha* beta* gamma + 1, beta* gamma - alpha, x[4],
                -beta, alpha, 1, x[5],
                0, 0, 0, 1;


        }*/
        
    }

    else{
        //parallelization part


        Eigen::MatrixXf A; // supposed to be ATA of the system according to paper

        Eigen::MatrixXf b;// supposed to be ATb according to paper

        Eigen::MatrixXf A_init = Eigen::MatrixXf::Zero(6, 6); // supposed to be ATA of the system according to paper

        Eigen::MatrixXf b_init = Eigen::MatrixXf::Zero(6, 1); // supposed to be ATb according to paper

        Vector3f* verticesSource;
        Vector3f* verticesPrevious;

        Vector3f* normalsSource;
        Vector3f* normalsPrevious;

        int width = curr_frame.width;
        int height = curr_frame.height;
        cudaMalloc((void**)&verticesSource, sizeof(Vector3f) * width * height);
        cudaMalloc((void**)&verticesPrevious, sizeof(Vector3f) * width * height);
        cudaMalloc((void**)&normalsSource, sizeof(Vector3f) * width * height);
        cudaMalloc((void**)&normalsPrevious, sizeof(Vector3f) * width * height);
       // cudaMalloc((void**)&estimatedPose, sizeof(Matrix4f));
       // cudaMalloc((void**)&transMatrixcur, sizeof(Matrix4f));
      //  cudaMalloc((void**)&K_calibration, sizeof(Matrix3f));
        cudaMalloc((void**)&A, sizeof(Matrix<float, 6, 6>) );
        cudaMalloc((void**)&b, sizeof(Matrix<float, 6, 1>) );

       // Vector3f curr_frame_V_k[190 * 120];
       // Vector3f  prev_frame_V_k[190 * 120];

       // Vector3f  curr_frame_N_k[190 * 120];
       // Vector3f  prev_frame_N_k[190 * 120];
        std::vector<Eigen::Vector3f> cr_fr_Vk = curr_frame.V_k;
        std::vector<Eigen::Vector3f> cr_fr_Nk = curr_frame.N_k;
        std::vector<Eigen::Vector3f> pr_fr_Vk = prev_frame.V_k;
        std::vector<Eigen::Vector3f> pr_fr_Nk = prev_frame.N_k;


        Eigen::MatrixXf A_final=Matrix<float, 6, 6>::Zero();

        Eigen::MatrixXf b_final= Matrix<float, 6, 1>::Zero();

      //  std::copy(cr_fr_Vk.begin(), cr_fr_Vk.end(), curr_frame_V_k);
      //  std::copy(cr_fr_Nk.begin(), cr_fr_Nk.end(), curr_frame_N_k);
//std::copy(pr_fr_Vk.begin(), pr_fr_Vk.end(), prev_frame_V_k);
       // std::copy(pr_fr_Nk.begin(), pr_fr_Nk.end(), prev_frame_N_k);

        cudaMemcpy(verticesSource, cr_fr_Vk.data(), sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(verticesPrevious, pr_fr_Vk.data(), sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(normalsSource, cr_fr_Nk.data(), sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(normalsPrevious, pr_fr_Nk.data(), sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
       // cudaMemcpy(estimatedPose, prev_frame.T_gk.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice);
      //  cudaMemcpy(&transMatrixcur, curr_frame.T_gk.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice);
        cudaMemcpy(&A, A_init.data(), sizeof(MatrixXf), cudaMemcpyHostToDevice);
        cudaMemcpy(&b, b_init.data(), sizeof(MatrixXf), cudaMemcpyHostToDevice);
       // cudaMemcpy(&K_calibration, prev_frame.K_calibration.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice);
        Matrix4f estimatedPose= prev_frame.T_gk;
        Matrix4f transMatrixcur = curr_frame.T_gk;
        Matrix3f K_calibration = prev_frame.K_calibration;
        float distance_threshold = 0.1f;
        float angle_threshold = 1.1f;
        int* count = 0;
        int* count_cpu = 0;
        cudaMalloc((void**)&count, sizeof(int));

        for (int i = 0; i < iterations; i++) 
        {

            correspondence_finder_cuda<<<1, 64 >>>(count,verticesSource, verticesPrevious, normalsSource, normalsPrevious,
                estimatedPose, transMatrixcur, K_calibration,
                width, height, distance_threshold, angle_threshold, A, b);
            cudaDeviceSynchronize();
            



        cudaMemcpy(A_final.data(), A.data(), sizeof(Matrix<float, 6, 6>), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_final.data(), b.data(), sizeof(Matrix<float, 6, 1>), cudaMemcpyDeviceToHost);
        cudaMemcpy(count_cpu, count, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << count_cpu << "\n";
        Eigen::Vector<float, 6> x = A_final.ldlt().solve(b_final); //ldlt because ATA not always Positive Definite

        float alpha = x[0];
        float beta = x[1];
        float gamma = x[2];


        T_gk_z << 1, alpha* beta - gamma, alpha* gamma + beta, x[3],
            gamma, alpha* beta* gamma + 1, beta* gamma - alpha, x[4],
            -beta, alpha, 1, x[5],
            0, 0, 0, 1;
        }
        cudaFree(verticesSource);
        cudaFree(verticesPrevious);
        cudaFree(normalsSource);
        cudaFree(normalsPrevious);

        cudaFree(&A);
        cudaFree(&b);

    }
    
    return T_gk_z;
    
    }

Eigen::Matrix4f ICP::pyramid_ICP(bool cuda){

    Eigen::Matrix4f T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[2], *prev_frame_pyramid -> Depth_Pyramid[2], 4, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[1], *prev_frame_pyramid -> Depth_Pyramid[1], 5, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[0], *prev_frame_pyramid -> Depth_Pyramid[0], 10, cuda);
    curr_frame_pyramid -> set_T_gk(T);

    return T;
}
