#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "ImplicitSurface.h"
#include "Volume.h"

// TODO: choose optimal truncation value
#define TRUNCATION 1.0
#define MAX_MARCHING_STEPS 10
#define EPSILON 0.05


struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0) -> why?
	Eigen::Vector3f position;
};

void writePointCloud(const std::string& filename, const std::vector<Vertex>& _vertices)
{
	std::ofstream file(filename);
	file << "OFF" << std::endl;
	file << _vertices.size() << " 0 0" << std::endl;
	for (unsigned int i = 0; i < _vertices.size(); ++i)
	{
		file << _vertices[i].position[0] << " " << _vertices[i].position[1] << " " << _vertices[i].position[2] << std::endl;
	}

}

int main()
{
	const auto imageWidth = 400; 
	const auto imageHeight = 400;

	Eigen::Matrix3f intrinsics; 
	Eigen::Vector3f cameraCenter(0.5f, 0.5f, -0.1f);
	intrinsics <<   525.0f, 0.0f, 319.5f,
					0.0f, 525.0f, 239.5f,
					0.0f, 0.0f, 1.0f;

	// Init implicit surface
	// Torus implicitTorus = Torus(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4, 0.1);
	Sphere implicit = Sphere(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4);
	// Fill spatial grid with distance to the implicit surface
	unsigned int mc_res = 50;
	Volume vol(Eigen::Vector3d(-0.1, -0.1, -0.1), Eigen::Vector3d(1.1, 1.1, 1.1), mc_res, mc_res, mc_res, 1);
	for (unsigned int x = 0; x < vol.getDimX(); x++)
	{
		for (unsigned int y = 0; y < vol.getDimY(); y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ(); z++)
			{
				Eigen::Vector3d p = vol.pos(x, y, z);
				double val = implicit.Eval(p);
				if (val < TRUNCATION)
					vol.set(x, y, z, val);
				else
					vol.set(x, y, z, TRUNCATION);
			}
		}
	}

	// Test function to check the point cloud writer
	// vol.writePointCloud("pointcloud.off");

	std::vector<Vertex> vertices;
	Eigen::Vector3f rayOrigin = vol.worldToGrid(cameraCenter);

	// Traverse the image pixel by pixel
	for (unsigned int j = 0; j < imageHeight; ++j)
	{
		for (unsigned int i = 0; i < imageWidth; ++i)
		{
			Eigen::Vector3f rayNext(float(i), float(j), 1.0f);
			Eigen::Vector3f rayNextCameraSpace = intrinsics.inverse() * rayNext;
			Eigen::Vector3f rayNextWorldSpace = rayNextCameraSpace + cameraCenter;
			Eigen::Vector3f rayNextGridSpace = vol.worldToGrid(rayNextWorldSpace);

			Eigen::Vector3f rayDir = rayNextGridSpace - rayOrigin;
			rayDir.normalize();

			// TODO: calculate first intersection with the volume (if exists)
			// First, let's try step size equal to one single voxel
			unsigned int step = 1;
		
			for (unsigned int s = 0; s < MAX_MARCHING_STEPS; ++s)
			{
				Eigen::Vector3f p = rayOrigin + step * rayDir;
				// Think carefully if this cast is correct or not
				if (!vol.outOfVolume(int(p[0]), int(p[1]), int(p[2])))
				{
					double dist = vol.get(p.cast<int>());
					if (dist < EPSILON)
					{	
						Vertex v = {
							p  // position
						};
						vertices.push_back(v);
						break;
					}
					step += 1;
					// std::cout << dist << std::endl;
				}
				else
				{
					std::cout << "OUT OF VOLUME" << std::endl;
					break;
				}
					
			}
		}
	}

	writePointCloud("pointcloud.off", vertices);

	return 0;
}