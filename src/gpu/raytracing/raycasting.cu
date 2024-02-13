#include <fstream>
#include "../../cpu/raytracing/Raycasting.h"
#include "../../cpu/tsdf/voxel.h"

/* CUDA */

__device__
Eigen::Vector3f worldToGrid(const Eigen::Vector3f& p, const Eigen::Vector3d& min, const Eigen::Vector3d& max, float ddx, float ddy, float ddz)
{
	Eigen::Vector3f coord(0.0, 0.0, 0.0);

	coord[0] = (p[0] - min[0]) / (max[0] - min[0]) / ddx;
	coord[1] = (p[1] - min[1]) / (max[1] - min[1]) / ddy;
	coord[2] = (p[2] - min[2]) / (max[2] - min[2]) / ddz;

	return coord;
}

__device__
Eigen::Vector3f gridToWorld(const Eigen::Vector3i& p, const Eigen::Vector3d& min, const Eigen::Vector3d& max, float ddx, float ddy, float ddz)
{
	Eigen::Vector3f coord(0.0f, 0.0f, 0.0f);

	coord[0] = min[0] + (max[0] - min[0]) * (float(p[0]) * ddx);
	coord[1] = min[1] + (max[1] - min[1]) * (float(p[1]) * ddy);
	coord[2] = min[2] + (max[2] - min[2]) * (float(p[2]) * ddz);

	return coord;
}

__device__
bool outOfVolume(int x, int y, int z, int dx, int dy, int dz)
{
	if (x < 0 || x > dx - 1 || y < 0 || y > dy - 1 || z < 0 || z > dz - 1)
			return true;
		return false;
}

__device__
bool computeNormal(Eigen::Vector3f& n, kinect_fusion::Voxel* tsdf, const Eigen::Vector3i& p, int dx, int dy, int dz)
{
	if (!outOfVolume(p[0] + 1, p[1], p[2], dx, dy, dz) && 
		!outOfVolume(p[0] - 1, p[1], p[2], dx, dy, dz) && 
		!outOfVolume(p[0], p[1] + 1, p[2], dx, dy, dz) && 
		!outOfVolume(p[0], p[1] - 1, p[2], dx, dy, dz) && 
		!outOfVolume(p[0], p[1], p[2] + 1, dx, dy, dz) && 
		!outOfVolume(p[0], p[1], p[2] - 1, dx, dy, dz))
	{
		double deltaX = tsdf[(p[0] + 1)*dy*dz + p[1]*dz + p[2]].tsdfValue - tsdf[(p[0] - 1)*dy*dz + p[1]*dz + p[2]].tsdfValue;
		double deltaY = tsdf[p[0]*dy*dz + (p[1] + 1)*dz + p[2]].tsdfValue - tsdf[p[0]*dy*dz + (p[1] - 1)*dz + p[2]].tsdfValue;
		double deltaZ = tsdf[p[0]*dy*dz + p[1]*dz + (p[2] + 1)].tsdfValue - tsdf[p[0]*dy*dz + p[1]*dz + (p[2] - 1)].tsdfValue;
		
		double gradX = -deltaX / 2.0f;
		double gradY = -deltaY / 2.0f;
		double gradZ = -deltaZ / 2.0f;

		n << gradX, gradY, gradZ;
		n.normalize();

		return true;
	}

	return false;
}

__global__
void castOneCuda(kinect_fusion::Voxel *tsdf, Eigen::Vector3f* dV_ks, Eigen::Vector3f* dN_ks, 
			     Eigen::Matrix3f rotation, Eigen::Vector3f translation, Eigen::Matrix3f intrinsics, Eigen::Matrix3f intrinsicsInv,
			     Eigen::Vector3d min, Eigen::Vector3d max, 
			     int width, int height, int dx, int dy, int dz, float ddx, float ddy, float ddz)
{
	const int i = threadIdx.x;
	const int j = blockIdx.x;

	if (i < width && j < height)
	{
		unsigned int marchingSteps = 20000;

		Eigen::Vector3f rayOrigin = worldToGrid(translation, min, max, ddx, ddy, ddz);
		Eigen::Vector3i rayOrigin_i = rayOrigin.cast<int>();
		Eigen::Vector3f rayNext(float(i), float(j), 1.0f);
		Eigen::Vector3f rayNextCameraSpace = intrinsicsInv * rayNext;
		Eigen::Vector3f rayNextWorldSpace = rotation * rayNextCameraSpace + translation;
		Eigen::Vector3f rayNextGridSpace = worldToGrid(rayNextWorldSpace, min, max, ddx, ddy, ddz);

		Eigen::Vector3f rayDir = rayNextGridSpace - rayOrigin;
		rayDir.normalize(); 

		float step = 7.0f;
		// printf("first step: %f \n", step);
		double prevDist = 0;
		bool intersected = false;

		for (unsigned int s = 0; s < marchingSteps; ++s)
		{
			Eigen::Vector3f p = rayOrigin + step * rayDir;
			Eigen::Vector3i pGrid = p.cast<int>();
			
			if (!outOfVolume(pGrid[0], pGrid[1], pGrid[2], dx, dy, dz))
			{
				intersected = true;
				double dist = tsdf[pGrid[0]*dy*dz + pGrid[1]*dz + pGrid[2]].tsdfValue;
				// printf("(%f, %f, %f, %f) \n", tsdf[pGrid[0]*dy*dz + pGrid[1]*dz + pGrid[2]].position[0], tsdf[pGrid[0]*dy*dz + pGrid[1]*dz + pGrid[2]].position[1], tsdf[pGrid[0]*dy*dz + pGrid[1]*dz + pGrid[2]].position[2], dist);
				// printf("%d \n", dist);
				if (!isnan(dist))
				{
					if (prevDist > 0 && dist <= 0 && s > 0)
					// 	break;
					// if (prevDist < 0 && dist >= 0 && s > 0)
					{	
						// Eigen::Vector3f interpolatedP = getInterpolatedIntersection(vol, rayOrigin, rayDir, step);
						Eigen::Vector3f n;
						if(computeNormal(n, tsdf, pGrid, dx, dy, dz))
						{
							Eigen::Vector3f pWorld = gridToWorld(pGrid, min, max, ddx, ddy, ddz);
							dV_ks[j * width + i] = pWorld;
							dN_ks[j * width + i] = n;
						}
					
						break;
					}
					prevDist = dist;
					// step += dist * 0.25f; 
					step += 1.0f;
					// printf("step s: %f", step);
				}
				else
					step += 7.0f;
				
			}
			else
			{	
				// If the ray has already intersected the volume and it is going out of boundaries
				if (intersected)
					break;
				// If not, conitnue traversing until finding intersection
				step += 7.0f;
			}						
		}
	}
}

/* DEFINITIONS FROM RAYCASTING.H */

Raycasting::Raycasting(kinect_fusion::VoxelGrid& _tsdf, const Eigen::Matrix3f& _extrinsics, const Eigen::Vector3f _cameraCenter): 
tsdf(_tsdf), extrinsincs(_extrinsics), cameraCenter(_cameraCenter)
{
	V_ks.resize(width * height);
	N_ks.resize(width * height);
    
	width = 640;
    height = 480;
    intrinsics <<   525.0f, 0.0f, 319.5f,
                    0.0f, 525.0f, 239.5f,
                    0.0f, 0.0f, 1.0f;

    // Initialize the vertices array which is going to copied to GPU
}

void Raycasting::writePointCloud(const std::string& filename)
{
    std::ofstream file(filename);

	for (unsigned int i = 0; i < width * height; ++i)
	{
		if (V_ks[i][0] != MINF && V_ks[i][1] != MINF && V_ks[i][2] != MINF)
		{
			file << "v " << V_ks[i][0] << " " << V_ks[i][1] << " " << V_ks[i][2] << std::endl;
			file << "vn " << N_ks[i][0] << " " << N_ks[i][0] << " " << N_ks[i][2] << std::endl;
		}
	}
}

void Raycasting::castAllCuda()
{
    // Vertex *vertices;
	Eigen::Vector3f* dV_ks;
	Eigen::Vector3f* dN_ks;

	// vertices = (Vertex*)malloc(width * height * sizeof(Vertex));

	cudaError_t cudaStatusVs = cudaMallocManaged(&dV_ks, width * height * sizeof(Eigen::Vector3f));
	if(cudaStatusVs != cudaSuccess)
	{
    	std::cout << "Problem in CudaMallocV_ks: " << cudaGetErrorString(cudaStatusVs) << std::endl;
    }
	
	cudaError_t cudaStatusNs = cudaMallocManaged(&dN_ks, width * height * sizeof(Eigen::Vector3f));
	if(cudaStatusNs != cudaSuccess)
	{
    	std::cout << "Problem in CudaMallocN_ks: " << cudaGetErrorString(cudaStatusNs) << std::endl;
    }

	castOneCuda <<<height, width>>> (tsdf.get_cu_grid(), dV_ks, dN_ks, 
							         extrinsincs, cameraCenter, intrinsics, intrinsics.inverse(),
			 				         tsdf.getMin(), tsdf.getMax(), 
			 				         width, height, tsdf.getDimX(), tsdf.getDimY(), tsdf.getDimZ(), tsdf.getSizeX(), tsdf.getSizeY(), tsdf.getSizeZ());
	cudaDeviceSynchronize();

	auto cudaCpy0 = cudaMemcpy(V_ks.data(), dV_ks, width * height * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
	if(cudaCpy0 != cudaSuccess)
	{
     	std::cout << "Problem in Copying dV_k from device: " << cudaGetErrorString(cudaCpy0) <<std::endl;
	}

	auto cudaCpy1 = cudaMemcpy(N_ks.data(), dN_ks, width * height * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
	if(cudaCpy1 != cudaSuccess)
	{
     	std::cout << "Problem in Copying dN_k from device: " << cudaGetErrorString(cudaCpy1) <<std::endl;
	}

	// writePointCloud(".\\cuda_out\\pointcloud.obj", vertices, width * height);

	cudaFree(dV_ks);
	cudaFree(dN_ks);
	// free(vertices);
	// freeCuda();
}

std::vector<Eigen::Vector3f> Raycasting::getVertices()
{
    return V_ks;
}

std::vector<Eigen::Vector3f> Raycasting::getNormals()
{
	return N_ks;
}

void Raycasting::freeCuda() {}

Raycasting::~Raycasting() { freeCuda(); }