#include "../../cpu/frame/Frame.h"
#include "../../cpu/frame/Frame_Pyramid.h"
#include <iostream>
#include <fstream>
#include <FreeImage.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#define MINF -std::numeric_limits<float>::infinity()

#define MAXTHRESHOLD 10

__global__
void apply_bilateral_cuda(float* depthMap, float* filteredImage, int diameter, double sigmaS, double sigmaR, int width, int height) {
	double filtered = 0;
	double wP = 0;
	int neighbor_x = 0;
	int neighbor_y = 0;
	int half = diameter / 2;
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	for (int i = 0; i < diameter; i++) {
		for (int j = 0; j < diameter; j++) {

			neighbor_x = id_x - (half - i);
			neighbor_y = id_y - (half - j);
			if (neighbor_x > 0 && neighbor_y > 0 && neighbor_x < height && neighbor_y < width)
			{
				if (depthMap[neighbor_x * width + neighbor_y] <= 0.0f || depthMap[neighbor_x * width + neighbor_y] == -INFINITY) {
					continue;
				}
				else
				{
					double N_r = exp(-(pow(sqrt(pow(depthMap[neighbor_x * width + neighbor_y] - depthMap[id_x * width + id_y], 2)), 2)) / pow(sigmaR, 2));

					double N_s = exp(-(pow(sqrt(pow(id_x - neighbor_x, 2) + pow(id_y - neighbor_y, 2)), 2)) / pow(sigmaS, 2));

					double w = N_s * N_r;
					filtered += depthMap[neighbor_x * width + neighbor_y] * w;
					wP = wP + w;
					//atomicAdd(&filtered, depthMap[neighbor_x * width + neighbor_y] * w);
					//atomicAdd(&wP, w);
				}
			}
		}
	}
	if (wP == 0.0) {
		filtered = 0;
	}

	else {
		filtered = filtered / wP;
	}

	filteredImage[id_x * width + id_y] = filtered;
}


__global__
void calculate_Vks_cuda(Eigen::Matrix3f K_i,
	Eigen::Vector3f* dV_k,
	float* Depth_k, int* dMk_0, int* dMk_1,
	int width, int height, bool online) {

	int id_x = threadIdx.x; // pixels per row
    int id_y = blockIdx.x;  // rows
	
	Eigen::Vector3f u_dot;

	 if(id_x < width && id_y < height) {
		int i = id_y * width + id_x;
		u_dot << id_x, id_y, 1;
		if (Depth_k[i] == -INFINITY || Depth_k[i] <= 0.0f) {
			dV_k[i] = Eigen::Vector3f(-INFINITY, -INFINITY, -INFINITY);
			dMk_0[i] = i;
		}
		else {
			// printf("I'm here %f", Depth_k[i]);
			if(online){
				dV_k[i] = Depth_k[i] * K_i * u_dot;
				dMk_1[i] = i;
			}
			else {
				dV_k[i] = Depth_k[i] * 255.0f * 255.0f / 5000.0f *K_i * u_dot;
				dMk_1[i] = i;
			}
			
		}
	}
}

__global__
void calculate_Nks_cuda(Eigen::Vector3f* dV_k,
	Eigen::Vector3f* dN_k,
	int width, int height)
{

	int id_x = threadIdx.x; // pixels per row
    int id_y = blockIdx.x;  // rows
	if(id_x < width - 1 && id_y < height - 1){
		
		int idx_vector = id_y * width + id_x;
		int idx_vector_right = id_y * width + id_x + 1;
		int idx_vector_down = (id_y + 1) * width + id_x;

		Eigen::Vector3f ans = (dV_k[idx_vector_right] - dV_k[idx_vector]).cross((dV_k[idx_vector_down] - dV_k[idx_vector]));
		ans.normalize();
		dN_k[idx_vector] = ans;
	}
	else if(id_x == width - 1 && id_y < height - 1) {

		int idx_vector = id_y * width + id_x;
		int idx_vector_left = id_y * width + id_x - 1;
		int idx_vector_down = (id_y + 1) * width + id_x;

		Eigen::Vector3f ans = (dV_k[idx_vector_left] - dV_k[idx_vector]).cross((dV_k[idx_vector_down] - dV_k[idx_vector]));
		ans.normalize();
		dN_k[idx_vector] = ans;
	}
	else if(id_x < width - 1 && id_y == height - 1) {

		int idx_vector = id_y * width + id_x;
		int idx_vector_right = id_y * width + id_x + 1;
		int idx_vector_up = (id_y - 1) * width + id_x;

		Eigen::Vector3f ans = (dV_k[idx_vector_right] - dV_k[idx_vector]).cross((dV_k[idx_vector_up] - dV_k[idx_vector]));
		ans.normalize();
		dN_k[idx_vector] = ans;
	}
	else if(id_x == width - 1 && id_y == height - 1) {

		int idx_vector = id_y * width + id_x;
		int idx_vector_left = id_y * width + id_x - 1;
		int idx_vector_up = (id_y - 1) * width + id_x;

		Eigen::Vector3f ans = (dV_k[idx_vector_left] - dV_k[idx_vector]).cross((dV_k[idx_vector_up] - dV_k[idx_vector]));
		ans.normalize();
		dN_k[idx_vector] = ans;
	}


}


std::vector<Eigen::Vector3f> Frame::calculate_Vks()
{
	V_k.resize(width * height);

	Eigen::Matrix3f K_i = K_calibration.inverse();

	Eigen::Vector3f* dV_k;

	int* dMk_1;
	int* dMk_0;

	//int* M_k1_new = new int[height * width];
	M_k0.resize(width * height);
	M_k1.resize(width * height);
	float* filtered_img_gpu;

	cudaError_t cudaStatus1 = cudaMallocManaged(&dV_k, height * width * sizeof(Eigen::Vector3f));
	if (cudaStatus1 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus1) << std::endl;
	};
	cudaError_t cudaStatus4 = cudaMallocManaged(&dMk_0, height * width * sizeof(float));
	if (cudaStatus4 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus4) << std::endl;
	};
	cudaError_t cudaStatus3 = cudaMallocManaged(&dMk_1, width * height * sizeof(float));
	if (cudaStatus3 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus3) << std::endl;
	};

	cudaError_t cudaStatus10 = cudaMallocManaged(&filtered_img_gpu, height * width * sizeof(float));
	if (cudaStatus10 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus10) << std::endl;
	};
	cudaError_t cudaStatus0 = cudaMemcpy(filtered_img_gpu, Depth_k, width * height * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus3 != cudaSuccess) {
		std::cout << "Problem in Copying1212: " << cudaGetErrorString(cudaStatus3) << std::endl;
	};

	int block_num = height;
    int thread_num = width;
	
	calculate_Vks_cuda <<< block_num, thread_num >>> (K_i, dV_k, filtered_img_gpu, dMk_0, dMk_1, width, height, true);
	// cudaDeviceSynchronize();

	cudaError_t cudaStatus2 = cudaMemcpy(V_k.data(), dV_k, width * height * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Problem in Copying1313: " << cudaGetErrorString(cudaStatus2) << std::endl;
	};
	cudaError_t cudaStatus5 = cudaMemcpy(M_k1.data(), dMk_1, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus5 != cudaSuccess) {
		std::cout << "Problem in Copying1414: " << cudaGetErrorString(cudaStatus5) << std::endl;
	};
	cudaError_t cudaStatus6 = cudaMemcpy(M_k0.data(), dMk_0, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus6 != cudaSuccess) {
		std::cout << "Problem in Copying1415: " << cudaGetErrorString(cudaStatus6) << std::endl;
	};
	cudaFree(dV_k);
	cudaFree(dMk_1);
	cudaFree(dMk_0);
	cudaFree(filtered_img_gpu);

	return V_k;
}

std::vector<Eigen::Vector3f>  Frame::calculate_Nks()
{
	N_k.resize(width * height);
	Eigen::Vector3f* dN_k;
	Eigen::Vector3f* V_k_array;

	cudaError_t cudaStatus1 = cudaMallocManaged(&dN_k, height * width * sizeof(Eigen::Vector3f));
	if (cudaStatus1 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus1) << std::endl;
	};
	cudaError_t cudaStatus2 = cudaMallocManaged(&V_k_array, height * width * sizeof(Eigen::Vector3f));
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus2) << std::endl;
	};
	cudaError_t cudaStatus3 = cudaMemcpy(V_k_array, V_k.data(), width * height * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
	if (cudaStatus3 != cudaSuccess) {
		std::cout << "Problem in Copying3: " << cudaGetErrorString(cudaStatus3) << std::endl;
	};

	int block_num = height;
    int thread_num = width;
	calculate_Nks_cuda <<<block_num, thread_num >>> (V_k_array, dN_k, width, height);
	cudaDeviceSynchronize();


	cudaError_t cudaStatus4 = cudaMemcpy(N_k.data(), dN_k, width * height * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
	if (cudaStatus4 != cudaSuccess) {
		std::cout << "Problem in Copying4: " << cudaGetErrorString(cudaStatus4) << std::endl;
	};

	cudaFree(dN_k);
	cudaFree(V_k_array);

	return N_k;
}

float* Frame::bilateralFilter_cu(int diameter, double sigmaS, double sigmaR) {
	float* depthMap = new float[height * width];
	float* filteredImage = new float[height * width];
	float* filteredImage_final = new float[height * width];


	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(height / threadsPerBlock.x, width / threadsPerBlock.y);

	cudaError_t cudaStatus1 = cudaMallocManaged(&filteredImage, height * width * sizeof(float));
	if (cudaStatus1 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus1) << std::endl;
	};
	cudaError_t cudaStatus4 = cudaMallocManaged(&depthMap, height * width * sizeof(float));
	if (cudaStatus4 != cudaSuccess) {
		std::cout << "Problem in memory allocation: " << cudaGetErrorString(cudaStatus4) << std::endl;
	};
	cudaError_t cudaStatus3 = cudaMemcpy(depthMap, Depth_k, width * height * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus3 != cudaSuccess) {
		std::cout << "Problem in Copying: " << cudaGetErrorString(cudaStatus3) << std::endl;
	};


	apply_bilateral_cuda <<< numBlocks, threadsPerBlock >>> (depthMap, filteredImage, diameter, sigmaS, sigmaR, width, height);
	cudaDeviceSynchronize();

	cudaError_t cudaStatus2 = cudaMemcpy(filteredImage_final, filteredImage, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Problem in Copying: " << cudaGetErrorString(cudaStatus2) << std::endl;
	};
	cudaFree(filteredImage);
	cudaFree(depthMap);
	//filteredImage_final = symmetricImageX(filteredImage_final, width, height);

	return filteredImage_final;
}

void Frame::save_off_format(const std::string& where_to_save) {

	std::ofstream OffFile(where_to_save);
	for (int j = 0; j < width * height; j++) {
		int i = M_k1[j];
		if (i == 0) {
			continue;
		}
		if (abs(V_k[i][0]) < MAXTHRESHOLD) {
			OffFile << "v " << V_k[i][0] << " " << V_k[i][1] << " " << V_k[i][2] << std::endl;
			if (!std::isnan(N_k[i][0]) && !std::isnan(N_k[i][1]) && !std::isnan(N_k[i][2])) {
				OffFile << "vn " << N_k[i][0] << " " << N_k[i][1] << " " << N_k[i][2] << std::endl;
			}
			else {
				OffFile << "vn " << 0 << " " << 0 << " " << 0 << std::endl;
			}
		}
	}
	OffFile.close();
}

Frame::Frame(FIBITMAP& dib, Eigen::Matrix4f T_gk, float sub_sampling_rate) :
	dib(FreeImage_ConvertToFloat(&dib)), T_gk(T_gk) {

	width = FreeImage_GetWidth(this->dib);
	height = FreeImage_GetHeight(this->dib);

	Depth_k = new float[width * height]; // have to rescale according to the data 

	Raw_k = (float*)FreeImage_GetBits(this->dib); // have to rescale according to the data 

	K_calibration << 525.0f / sub_sampling_rate, 0.0f, 319.5f / sub_sampling_rate,
		0.0f, 525.0f / sub_sampling_rate, 239.5f / sub_sampling_rate,
		0.0f, 0.0f, 1.0f;
}

Frame::Frame(std::vector<float> depthMap, Eigen::Matrix4f T_gk, Eigen::Matrix3f K, int width, int height, float sub_sampling_rate) :
	dib(nullptr), T_gk(T_gk), K_calibration(K), Raw_k(depthMap.data()), width(width), height(height) {
	// live sensor constructor
	K(0,0) /= sub_sampling_rate;
	K(0,2) /= sub_sampling_rate;
	K(1,1) /= sub_sampling_rate;
	K(1,2) /= sub_sampling_rate;
	
	Depth_k = new float[width * height]; // have to rescale according to the data 
	for(int i = 0; i < width*height; i++){
		Depth_k[i] = depthMap[i];
	}
}

Frame::Frame(const char* image_dir, Eigen::Matrix4f T_gk, float sub_sampling_rate) :
	dib(FreeImage_ConvertToFloat(FreeImage_Load(FreeImage_GetFileType(image_dir), image_dir))) {

	// FreeImage_Initialise();

	width = FreeImage_GetWidth(this->dib);
	height = FreeImage_GetHeight(this->dib);

	Depth_k = new float[width * height]; // have to rescale according to the data 

	Raw_k = (float*)FreeImage_GetBits(this->dib); // have to rescale according to the data 

	K_calibration << 525.0f / sub_sampling_rate, 0.0f, 319.5f / sub_sampling_rate,
		0.0f, 525.0f / sub_sampling_rate, 239.5f / sub_sampling_rate,
		0.0f, 0.0f, 1.0f;

	this->T_gk = T_gk;

	// FreeImage_DeInitialise();
}

Frame::~Frame() {
	
	if (dib != nullptr) { 
		FreeImage_Unload(dib);
		dib = nullptr;
	}
	if (Depth_k != nullptr) { 
		delete[] Depth_k;
		Depth_k = nullptr;
	}
	if(Raw_k != nullptr){
//		FreeImage_Unload((FITBITMAP *)Raw_k);
		Raw_k = nullptr;
	}
	if(filtered_dib != nullptr){
		FreeImage_Unload(filtered_dib);
		filtered_dib = nullptr;
	}

}

void Frame::process_image(float sigma_r, float sigma_s, int filter_size, bool apply_bilateral) {
	
	if (apply_bilateral) {
		Depth_k = bilateralFilter_cu(3, 3.0, 0.01);
	}
	else
	{
		Depth_k = Raw_k;
	}
	// cuda
	calculate_Vks();
	calculate_Nks();
	//save_off_format("C:/Users/yigitavci/Desktop/TUM_DERS/Semester_2/3D_Scanning/KinectFusion-Cool-Edition/scene2_cudaa_nofilter.obj");

}

Frame::Frame(std::vector<Eigen::Vector3f> V_gks, std::vector<Eigen::Vector3f> N_gks, Eigen::Matrix4f T_gk, int width, int height):
width(width), height(height), T_gk(T_gk), V_gk(V_gks), N_gk(N_gks){
    K_calibration  <<  525.0f , 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;
	transformed = true;
    
}

void Frame::save_G_off_format(const std::string & where_to_save)
{
        std::ofstream OffFile(where_to_save);
        this -> apply_G_transform();
        for(unsigned int i = 0; i < width * height; ++i){
            if(abs(V_gk[i][0]) < MAXTHRESHOLD){
                if (V_gk[i][0] != MINF)
                {
                    OffFile << "v " << V_gk[i][0] << " " << V_gk[i][1] << " " << V_gk[i][2] << std::endl; 
                    if(!std::isnan(N_gk[i][0]) && !std::isnan(N_gk[i][1]) && !std::isnan(N_gk[i][2])){
                        OffFile << "vn " << N_gk[i][0] << " " << N_gk[i][1] << " " << N_gk[i][2] << std::endl;
                    }
                    else{
                        OffFile << "vn " << 0 << " " << 0 << " " << 0 << std::endl;
                    } 
                }
            }
        }
        OffFile.close();
    }
