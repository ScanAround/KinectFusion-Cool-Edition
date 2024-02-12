#include "../../cpu/icp/ICP.h"
#include "../../cpu/frame/Frame_Pyramid.h"
#include <FreeImage.h>
#include <fstream>
#include <chrono>

__device__
void update_lower_triangle(float *A_jT, float* A){
    //column major update of lower triangle 
    int i = 0;
    for(int col = 0; col < 6; ++col){
        for(int row = col; row < 6; ++row){
            A[i] = A_jT[row] * A_jT[col];
            i++;
        }
    }
}

//might wanna define it in header then reference here and tsdf
__device__ 
Eigen::Vector2i vec_to_pixel(
    const Eigen::Vector3f vec,
    Eigen::Matrix3f R_prev, Eigen::Vector3f t_prev,
    Eigen::Matrix3f K_calibration,
    int width, int height){

  Eigen::Vector3f vec_camera_frame = R_prev * vec + t_prev;
  
  Eigen::Vector3f u_dot = (K_calibration * vec_camera_frame) / vec_camera_frame[2];

  Eigen::Vector2i u;
  if(u_dot[0] >= 0 
  && u_dot[0] < width 
  && u_dot[1] >= 0 
  && u_dot[1] < height){
      u << int(u_dot[0]), int(u_dot[1]);
  }
  else{
      u << 0,0 ;
  }
  return u;
}

//maybe I should do shared memeory!
__global__ 
void A_b_finder_block(
    Eigen::Matrix<float, 21, 1> *dA_arr, Eigen::Matrix<float, 6, 1> *db_arr,
    Eigen::Matrix3f R_it, Eigen::Vector3f t_it, 
    Eigen::Matrix3f R_prev_i, Eigen::Vector3f t_prev_i, 
    Eigen::Vector3f *curr_V_k, Eigen::Vector3f *curr_N_k,
    Eigen::Vector3f *prev_V_gk, Eigen::Vector3f *prev_N_gk,
    Eigen::Matrix3f K, 
    int width, int height,
    float d_thresh, float a_thresh){

        int id_x = threadIdx.x; // pixels per row
        int id_y = blockIdx.x;  // rows

        if(id_x < width && id_y < height){
            int i = id_y * width + id_x;
            // transforming the current frame to the current pose iteration
            auto curr_V_k_t = R_it * curr_V_k[i] + t_it; //converting V_k to V_gk_i
            auto curr_N_k_t = R_it * curr_N_k[i]; //converting N_k to N_gk_i

            //projecting back to previous frame pixels
            Eigen::Vector2i pixel = vec_to_pixel(curr_V_k_t, R_prev_i, t_prev_i, K, width, height);
            int idx_in_prev = pixel[1]*width + pixel[0];           
            
            if(idx_in_prev < height * width){
                // checking if normals are valid in previous frame
                if(!isnan(prev_N_gk[idx_in_prev][0]) 
                && !isnan(prev_N_gk[idx_in_prev][1]) 
                && !isnan(prev_N_gk[idx_in_prev][2])
                && !isnan(prev_V_gk[idx_in_prev][0])
                && !isnan(prev_V_gk[idx_in_prev][1])
                && !isnan(prev_V_gk[idx_in_prev][2])
                && !isnan(curr_V_k_t[0])){
                    if((curr_V_k_t - prev_V_gk[idx_in_prev]).norm() <= d_thresh){
                        if(abs(curr_N_k_t.dot(prev_N_gk[idx_in_prev])) >= a_thresh){
                            Eigen::Matrix<float, 3, 1, Eigen::ColMajor> s_i = curr_V_k_t; // already transformed
                            Eigen::Matrix<float, 3, 1, Eigen::ColMajor> d_i = prev_V_gk[idx_in_prev];
                            Eigen::Matrix<float, 3, 1, Eigen::ColMajor> n_i = prev_N_gk[idx_in_prev];
                            
                            Eigen::Matrix<float, 6, 1, Eigen::ColMajor> A_jT;
                            A_jT << s_i.cross(n_i), n_i;

                            Eigen::Matrix<float, 21, 1, Eigen::ColMajor> _A;
                            update_lower_triangle(A_jT.data(), _A.data());

                            // might want to want to change dA_arr from eigen matrix to float array
                            // atomicAdd to stop race sync problems
                            // __syncthreads();
                            for(int element = 0; element < 21 ; ++element){
                                atomicAdd(&dA_arr[id_y](element), _A.data()[element]);
                                if (element < 6) atomicAdd(&db_arr[id_y](element), A_jT.data()[element] * (n_i.dot(d_i) - n_i.dot(s_i)));
                            }                 
                        }
                    }
                }
            }
        }
    }

__global__ 
void sum_over_blocks(
    Eigen::Matrix<float, 21, 1> *dA_arr, Eigen::Matrix<float, 6, 1> *db_arr, int size,
    Eigen::Matrix<float, 21, 1> *dA_sum, Eigen::Matrix<float, 6, 1> *db_sum){
        
        int idx = threadIdx.x;
        // printf("ith element %f", dA_arr[idx](0));
        if(idx < size){
            for(int i = 0; i < 21; ++i){
                atomicAdd(&dA_sum[0](i), dA_arr[idx](i));
                if (i < 6) atomicAdd(&db_sum[0](i), db_arr[idx](i));
            }
        }
}

Eigen::Matrix4f ICP::point_to_plane_solver(Frame & curr_frame, Frame & prev_frame, int iterations, bool cuda){
    
    // source is the live frame F_k and the prev_frame is the ray-casted previous frame F_k-1
    Eigen::Matrix4f T_gk_z = curr_frame.T_gk;
    Eigen::Matrix4f T_gk_z_temp;
    prev_frame.apply_G_transform();
    
    Eigen::Vector3f *curr_V_k;
    Eigen::Vector3f *curr_N_k;
    Eigen::Vector3f *prev_V_gk;
    Eigen::Vector3f *prev_N_gk;
    
    cudaError_t cudaStatus0 = cudaMallocManaged(&curr_V_k, sizeof(Eigen::Vector3f) * curr_frame.V_k.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in cudaMallocManaged curr_V_k: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(curr_V_k, curr_frame.V_k.data(), sizeof(Eigen::Vector3f) * curr_frame.V_k.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of curr_V_k: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }

    cudaStatus0 = cudaMallocManaged(&curr_N_k, sizeof(Eigen::Vector3f) * curr_frame.N_k.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in cudaMallocManaged curr_N_k: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(curr_N_k, curr_frame.N_k.data(), sizeof(Eigen::Vector3f) * curr_frame.N_k.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of curr_N_k: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    
    cudaStatus0 = cudaMallocManaged(&prev_V_gk, sizeof(Eigen::Vector3f) * prev_frame.V_gk.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in cudaMallocManaged prev_V_gk: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(prev_V_gk, prev_frame.V_gk.data(), sizeof(Eigen::Vector3f) * prev_frame.V_gk.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of prev_V_gk: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }

    cudaStatus0 = cudaMallocManaged(&prev_N_gk, sizeof(Eigen::Vector3f) * prev_frame.N_gk.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in cudaMallocManaged prev_N_gk: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(prev_N_gk, prev_frame.N_gk.data(), sizeof(Eigen::Vector3f) * prev_frame.N_gk.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of prev_N_gk: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }

    for(int i = 0; i < iterations; i++){
        
        Eigen::Matrix<float, 6, 6> A = Eigen::MatrixXf::Zero(6, 6);
        Eigen::Matrix<float, 21, 1> LA = Eigen::MatrixXf::Zero(21, 1);
        Eigen::Matrix<float, 6, 1> b = Eigen::MatrixXf::Zero(6, 1);
        // const int tile_dim = 8;
        // dim3 thread_num(tile_dim, tile_dim); 
        // dim3 block_num(curr_frame.width/tile_dim, curr_frame.height/tile_dim);

        int block_num = curr_frame.height;
        int thread_num = curr_frame.width;

        Eigen::Matrix<float, 21, 1>* dA_arr;
        Eigen::Matrix<float, 21, 1>* dA_sum;
        Eigen::Matrix<float, 6, 1>* db_arr;
        Eigen::Matrix<float, 6, 1>* db_sum;
        cudaMallocManaged(&dA_arr, sizeof(Eigen::Matrix<float, 21, 1>) * block_num);
        cudaMallocManaged(&dA_sum, sizeof(Eigen::Matrix<float, 21, 1>));
        cudaMallocManaged(&db_arr, sizeof(Eigen::Matrix<float, 6, 1>) * block_num);
        cudaMallocManaged(&db_sum, sizeof(Eigen::Matrix<float, 6, 1>));

        // Copy the temporary storage to the device
        cudaMemcpy(dA_sum, &A, sizeof(Eigen::Matrix<float, 21, 1>), cudaMemcpyHostToDevice);
        cudaMemcpy(db_sum, &b, sizeof(Eigen::Matrix<float, 6, 1>), cudaMemcpyHostToDevice);

        A_b_finder_block <<<block_num, thread_num>>>(
            dA_arr, db_arr, 
            T_gk_z.block(0,0,3,3), T_gk_z.block(0,3,3,1),
            prev_frame.T_gk.inverse().block(0,0,3,3), prev_frame.T_gk.inverse().block(0,3,3,1),
            curr_V_k, curr_N_k,
            prev_V_gk, prev_N_gk,
            prev_frame.K_calibration, 
            prev_frame.width, prev_frame.height, 
            distance_threshold, angle_threshold
        );
        cudaDeviceSynchronize();
        
        sum_over_blocks <<<1, block_num>>>(
            dA_arr, db_arr, curr_frame.width * curr_frame.height,   
            dA_sum, db_sum
        );
        cudaDeviceSynchronize();


        cudaStatus0 = cudaMemcpy(&LA, dA_sum, sizeof(Eigen::Matrix<float, 21, 1>), cudaMemcpyDeviceToHost);
        // __syncthreads();
        if(cudaStatus0 != cudaSuccess){
            std::cout << "Problem in Cuda Copy of dA to A: " << cudaGetErrorString(cudaStatus0) << std::endl;
        }
        
        cudaStatus0 = cudaMemcpy(&b, db_sum, sizeof(Eigen::Matrix<float, 6, 1>), cudaMemcpyDeviceToHost);
        if(cudaStatus0 != cudaSuccess){
            std::cout << "Problem in Cuda Copy of db to b: " << cudaGetErrorString(cudaStatus0) << std::endl;
        }

        cudaStatus0 = cudaGetLastError();
        if (cudaStatus0 != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(cudaStatus0) << std::endl;
        }

        int i_temp = 0;
        for(int col = 0; col < 6; ++col){
            for(int row = col; row < 6; ++row){
                A(row, col) = LA[i_temp];
                i_temp++;
            }
        }
        Eigen::Vector<float, 6> x = A.ldlt().solve(b); //ldlt because ATA not always Positive Definite
        
        
        float alpha = x[0];
        float beta = x[1];
        float gamma = x[2];


        T_gk_z_temp <<      1 ,  alpha*beta - gamma , alpha*gamma + beta , x[3],
                gamma ,  alpha*beta*gamma + 1   ,  beta*gamma - alpha , x[4],
                -beta , alpha ,   1   , x[5],
                    0    ,  0    ,   0   ,  1  ; 
        
        T_gk_z = T_gk_z_temp * T_gk_z;

        // curr_frame_pyramid->set_T_gk(T_gk_z);
        // curr_frame_pyramid -> Depth_Pyramid[0]->save_G_off_format("iter" + std::to_string(i) + ".obj");
        cudaFree(dA_arr);
        cudaFree(dA_sum);
        cudaFree(db_arr);
        cudaFree(db_sum);
    }

    cudaFree(curr_V_k);
    cudaFree(curr_N_k);
    cudaFree(prev_V_gk);
    cudaFree(prev_N_gk);
        
    return T_gk_z;
    
}

Eigen::Matrix4f ICP::pyramid_ICP(bool cuda){
    std::cout << "Registering 3rd Frame Pyramid Level" << std::endl;
    Eigen::Matrix4f T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[2], *prev_frame_pyramid -> Depth_Pyramid[2], 4, cuda);
    curr_frame_pyramid -> set_T_gk(T);

    // curr_frame_pyramid -> Depth_Pyramid[2]->save_off_format("outputs/point_clouds/pc_level2.obj");

    std::cout << "Registering 2nd Frame Pyramid Level" << std::endl;
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[1], *prev_frame_pyramid -> Depth_Pyramid[1], 4, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    // curr_frame_pyramid -> Depth_Pyramid[1]->save_off_format("outputs/point_clouds/pc_level1.obj");
    
    std::cout << "Registering 1st Frame Pyramid Level" << std::endl;
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[0], *prev_frame_pyramid -> Depth_Pyramid[0], 10, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    // curr_frame_pyramid -> Depth_Pyramid[0]->save_off_format("outputs/point_clouds/pc_level0.obj");

    return T;
}

// int main(){

//     FreeImage_Initialise();
//     const char* depth_map_dir_1 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
//     const char* depth_map_dir_2 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.226738.png";

//     Frame_Pyramid* frame1 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_1), depth_map_dir_1));
//     frame1->Depth_Pyramid[0]->save_off_format("scene1.obj");

//     Frame_Pyramid* frame2 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_2), depth_map_dir_2));
//     frame2->Depth_Pyramid[0]->save_off_format("scene2.obj");

//     auto start = std::chrono::high_resolution_clock::now();
//     std::cout << "starting timer" << std::endl;
//     ICP icp(*frame1, *frame2, 0.05f, 0.5f);
//     auto T = icp.pyramid_ICP(false);
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
//     std::cout << "time for execution: " << duration << std::endl; 
    
//     std::cout << T << std::endl;

//     std::vector<Eigen::Vector3f> V_tk;
//     frame1->Depth_Pyramid[0]->apply_transform(T , V_tk);

//     std::ofstream OffFile("transformed_scene_1.obj");
//     for(auto V : V_tk){
//         OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
//     }
//     OffFile.close();
// }