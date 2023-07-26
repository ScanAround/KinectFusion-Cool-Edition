#include "../../cpu/icp/ICP.h"
#include "../../cpu/frame/Frame_Pyramid.h"
#include <FreeImage.h>
#include <fstream>
#include <chrono>

__device__
void update_lower_triangle(float *A_jT, float* A){
    //column major update of lower triangle 
    for(int col = 0; col < 6; ++col){
        for(int row = col; row < 6; ++row){
            A[col * 6 + row] = A_jT[row] * A_jT[col];
        }
    }
}

//might wanna define it in header then reference here and tsdf
__device__ 
Eigen::Vector2i vec_to_pixel(
    const Eigen::Vector3f vec,
    Eigen::Matrix3f R_i, Eigen::Vector3f t_i,
    Eigen::Matrix3f K_calibration,
    int width, int height){

  Eigen::Vector3f vec_camera_frame = R_i * vec + t_i;
  
  Eigen::Vector3f u_dot = (K_calibration * vec_camera_frame) / vec_camera_frame[2];

  Eigen::Vector2i u;
  if(u_dot[0] >= 0 
  && u_dot[0] <= width 
  && u_dot[1] >= 0 
  && u_dot[1] <= height){
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
    Eigen::Matrix3f R_curr, Eigen::Vector3f t_curr, 
    Eigen::Matrix3f R_prev, Eigen::Vector3f t_prev, 
    Eigen::Vector3f *curr_V_k, Eigen::Vector3f *curr_N_k,
    Eigen::Vector3f *prev_V_gk, Eigen::Vector3f *prev_N_gk,
    Eigen::Matrix3f K, 
    int width, int height,
    int d_thresh, int a_thresh){

        int id_x = blockIdx.x; // each column of the image given to a block in cuda
        int id_y = threadIdx.x; // each pixel in every column given to a thread

        if(id_x < width && id_y < height){
            int i = id_y * width + id_x;

            // transforming the current frame to the current pose iteration
            curr_V_k[i] = R_curr * curr_V_k[i] + t_curr;
            curr_N_k[i] = R_curr * curr_N_k[i];

            //projecting back to previous frame pixels
            Eigen::Vector2i pixel = vec_to_pixel(curr_V_k[i], R_prev, t_prev, K, width, height);
            int idx_in_prev = pixel[1]*width + pixel[0];

            // normals aren't calculated in last row so check if pixel is before last row
            if(idx_in_prev < (height-1) * width){
                // checking if normals are valid in previous frame
                if(!std::isnan(prev_N_gk[idx_in_prev][0])){
                    // checking if distances within threshold
                    if((curr_V_k[i] - prev_V_gk[idx_in_prev]).norm() <= d_thresh){
                        // checking if angles within threshold
                        if(curr_N_k[i].dot(prev_N_gk[idx_in_prev]) >= a_thresh){

                            Eigen::Vector3f s_i = R_curr * curr_V_k[i] + t_curr;
                            Eigen::Vector3f d_i = prev_V_gk[idx_in_prev];
                            Eigen::Vector3f n_i = prev_N_gk[idx_in_prev];
                            Eigen::Matrix<float, 6, 1, Eigen::ColMajor> A_jT;
                            A_jT << s_i.cross(n_i), n_i;
                            
                            Eigen::Matrix<float, 21, 1, Eigen::ColMajor> _A;
                            // A.selfadjointView<Eigen::Lower>().rankUpdate(A_jT);
                            update_lower_triangle(A_jT.data(), _A.data());
                            dA_arr[id_x] += _A;
                            db_arr[id_x] += A_jT * (n_i.dot(d_i) - n_i.dot(s_i));
                            __syncthreads();
                        }
                    }
                }
            }
        }
    }

Eigen::Matrix4f ICP::point_to_plane_solver(Frame & curr_frame, Frame & prev_frame, int iterations, bool cuda){
    
    // source is the live frame F_k and the prev_frame is the ray-casted previous frame F_k-1
    Eigen::Matrix4f T_gk_z = curr_frame.T_gk;
    prev_frame.apply_G_transform();
    
    Eigen::Vector3f *curr_V_k;
    Eigen::Vector3f *curr_N_k;
    Eigen::Vector3f *prev_V_gk;
    Eigen::Vector3f *prev_N_gk;
    
    cudaError_t cudaStatus0 = cudaMalloc(&curr_V_k, sizeof(Eigen::Vector3f) * curr_frame.V_k.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in CudaMalloc curr_V_k: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(curr_V_k, curr_frame.V_k.data(), sizeof(Eigen::Vector3f) * curr_frame.V_k.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of dA: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }

    cudaStatus0 = cudaMalloc(&curr_N_k, sizeof(Eigen::Vector3f) * curr_frame.N_k.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in CudaMalloc curr_N_k: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(curr_N_k, curr_frame.N_k.data(), sizeof(Eigen::Vector3f) * curr_frame.N_k.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of dA: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    
    cudaStatus0 = cudaMalloc(&prev_V_gk, sizeof(Eigen::Vector3f) * prev_frame.V_gk.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in CudaMalloc prev_V_gk: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(prev_V_gk, prev_frame.V_gk.data(), sizeof(Eigen::Vector3f) * prev_frame.V_gk.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of dA: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }

    cudaStatus0 = cudaMalloc(&prev_N_gk, sizeof(Eigen::Vector3f) * prev_frame.N_gk.size());
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in CudaMalloc prev_N_gk: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }
    cudaStatus0 = cudaMemcpy(prev_N_gk, prev_frame.N_gk.data(), sizeof(Eigen::Vector3f) * prev_frame.N_gk.size(), cudaMemcpyHostToDevice);
    if(cudaStatus0 != cudaSuccess){
        std::cout << "Problem in Cuda Copy of dA: " << cudaGetErrorString(cudaStatus0) << std::endl;
    }

    for(int i = 0; i < iterations; i++){
        
        Eigen::Matrix<float, 6, 6> A = Eigen::MatrixXf::Zero(6, 6);
        Eigen::Matrix<float, 6, 1> b = Eigen::MatrixXf::Zero(6, 1);
        int block_num = curr_frame.width;
        int thread_num = curr_frame.height;

        Eigen::Matrix<float, 6, 6>* dA;
        Eigen::Matrix<float, 6, 1>* db;
        cudaMalloc(&dA, sizeof(Eigen::Matrix<float, 6, 6>));
        cudaMalloc(&db, sizeof(Eigen::Matrix<float, 6, 1>));

        // Copy the temporary storage to the device
        cudaMemcpy(dA, &A, sizeof(Eigen::Matrix<float, 6, 6>), cudaMemcpyHostToDevice);
        cudaMemcpy(db, &b, sizeof(Eigen::Matrix<float, 6, 1>), cudaMemcpyHostToDevice);


        A_b_finder_block <<<block_num, thread_num>>>(
            dA, db, 
            curr_frame.T_gk.block(0,0,3,3), curr_frame.T_gk.block(0,3,3,1),
            prev_frame.T_gk.block(0,0,3,3), prev_frame.T_gk.block(0,3,3,1),
            curr_V_k, curr_N_k,
            prev_V_gk, prev_N_gk,
            prev_frame.K_calibration, 
            prev_frame.width, prev_frame.height, 
            distance_threshold, angle_threshold
        );
        cudaDeviceSynchronize();
        // for()
        cudaStatus0 = cudaMemcpy(&A, dA, sizeof(Eigen::Matrix<float, 6, 6>) * block_num, cudaMemcpyDeviceToHost);
        if(cudaStatus0 != cudaSuccess){
            std::cout << "Problem in Cuda Copy of dA to A: " << cudaGetErrorString(cudaStatus0) << std::endl;
        }
        
        cudaStatus0 = cudaMemcpy(&A, db, sizeof(Eigen::Matrix<float, 6, 1>) * block_num, cudaMemcpyDeviceToHost);
        if(cudaStatus0 != cudaSuccess){
            std::cout << "Problem in Cuda Copy of db to b: " << cudaGetErrorString(cudaStatus0) << std::endl;
        }
        
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

        Eigen::Vector<float, 6> x = A.ldlt().solve(b); //ldlt because ATA not always Positive Definite
        
        
        float alpha = x[0];
        float beta = x[1];
        float gamma = x[2];


        T_gk_z <<      1 ,  alpha*beta - gamma , alpha*gamma + beta , x[3],
                gamma ,  alpha*beta*gamma + 1   ,  beta*gamma - alpha , x[4],
                -beta , alpha ,   1   , x[5],
                    0    ,  0    ,   0   ,  1  ; 
        
        
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
    
    std::cout << "Registering 2nd Frame Pyramid Level" << std::endl;
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[1], *prev_frame_pyramid -> Depth_Pyramid[1], 5, cuda);
    curr_frame_pyramid -> set_T_gk(T);
    
    std::cout << "Registering 1st Frame Pyramid Level" << std::endl;
    T = this -> point_to_plane_solver(*curr_frame_pyramid -> Depth_Pyramid[0], *prev_frame_pyramid -> Depth_Pyramid[0], 10, cuda);
    curr_frame_pyramid -> set_T_gk(T);

    return T;
}

int main(){

    FreeImage_Initialise();
    const char* depth_map_dir_1 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    const char* depth_map_dir_2 = "/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.194330.png";

    Frame_Pyramid* frame1 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_1), depth_map_dir_1));
    frame1->Depth_Pyramid[0]->save_off_format("scene1.obj");

    Frame_Pyramid* frame2 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir_2), depth_map_dir_2));
    frame2->Depth_Pyramid[0]->save_off_format("scene2.obj");

    auto start = std::chrono::high_resolution_clock::now();
    ICP icp(*frame1, *frame2, 0.1f, 1.1f);
    auto T = icp.pyramid_ICP(false);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "time for execution: " << duration << std::endl; 
    
    std::cout << T;

    std::vector<Eigen::Vector3f> V_tk;
    frame1->Depth_Pyramid[0]->apply_transform(T , V_tk);

    std::ofstream OffFile("/home/amroabuzer/Desktop/KinectFusion/KinectFusion-Cool-Edition/transformed_scene_1.obj");
    for(auto V : V_tk){
        OffFile << "v " << V[0] << " " << V[1] << " " << V[2] << std::endl; 
    }
    OffFile.close();
}