#define _USE_MATH_DEFINES  // to access M_PI macro from math.h

#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include "ImplicitSurface.h"
#include "Volume.h"

// TODO: choose optimal truncation value
#define TRUNCATION 1.0
#define MAX_MARCHING_STEPS 5
#define EPSILON 0.1


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
	const auto imageWidth = 640; 
	const auto imageHeight = 480;

	Eigen::Matrix3f intrinsics; 
	intrinsics <<   525.0f, 0.0f, 319.5f,
					0.0f, 525.0f, 239.5f,
					0.0f, 0.0f, 1.0f;

	Eigen::Vector3f cameraCenter(2.0f, 2.0f, -2.0f);

	// Define rotation with Euler angles
	float alpha = 30 * (M_PI / 180);  // x
	float beta = -30 * (M_PI / 180);   // y
	float gamma = 0 * (M_PI / 180);  // z
	Eigen::Matrix3f rotationX;
	Eigen::Matrix3f rotationY;
	Eigen::Matrix3f rotationZ;

	rotationX << 1.0f, 0.0f, 0.0f, 
				 0.0f, cos(alpha), -sin(alpha), 
				 0.0f, sin(alpha), cos(alpha);
	rotationY << cos(beta), 0.0f, sin(beta),
				 0.0f, 1.0f, 0.0f, 
				 -sin(beta), 0.0f, cos(beta);
	rotationZ << cos(gamma), -sin(gamma), 0.0f,
				 sin(gamma), cos(gamma), 0.0f,
				 0.0f, 0.0f, 1.0f;

	Eigen::Matrix3f rotation = rotationZ * rotationY * rotationX;

	std::vector<Vertex> vertices;

	Vertex c = {
		cameraCenter
	};
	vertices.push_back(c);

	// Init implicit surface
	// Torus implicitTorus = Torus(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4, 0.1);
	Sphere implicit = Sphere(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4);
	// Fill spatial grid with distance to the implicit surface
	unsigned int mc_res = 300;
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

				if ( x % 10 == 0 && y % 10 == 0 && z % 10 == 0)
				{
					Vertex v = {
							p.cast<float>()  // position
						};
					vertices.push_back(v);
				}
					
			}
		}
	}

	// Test function to check the point cloud writer
	// vol.writePointCloud("pointcloud.off");

	Eigen::Vector3f rayOrigin = vol.worldToGrid(cameraCenter);

	// Traverse the image pixel by pixel
	for (unsigned int j = 0; j < imageHeight; ++j)
	{
		for (unsigned int i = 0; i < imageWidth; ++i)
		{
			Eigen::Vector3f rayNext(float(i), float(j), 1.0f);
			Eigen::Vector3f rayNextCameraSpace = intrinsics.inverse() * rayNext;
			Eigen::Vector3f rayNextWorldSpace = rotation * rayNextCameraSpace + cameraCenter;
			Eigen::Vector3f rayNextGridSpace = vol.worldToGrid(rayNextWorldSpace);

			//Eigen::Vector3f rayDir = rayNextGridSpace - rayOrigin;
			Eigen::Vector3f rayDir = rayNextWorldSpace - cameraCenter;
			rayDir.normalize();

			// TODO: calculate first intersection with the volume (if exists)
			// First, let's try step size equal to one single voxel
			float step = 1.0f;
			double prevDist = 0;
			bool intersected = false;
			if (j % 10 == 0 && i % 10 == 0)
			{
				for (unsigned int s = 0; s < MAX_MARCHING_STEPS; ++s)
				{
					Eigen::Vector3f p = cameraCenter + step * rayDir;
					Vertex v = {
								p  // position
							};
					vertices.push_back(v);
					step += .5f;
					
					/* // Think carefully if this cast is correct or not
					if (!vol.outOfVolume(int(p[0]), int(p[1]), int(p[2])))
					{
						intersected = true;
						double dist = vol.get(p.cast<int>());
						if (prevDist > 0 && dist <=0 && s > 0)
						{	
							Vertex v = {
								p  // position
							};
							vertices.push_back(v);
							break;
						}
						prevDist = dist;
						step += 1.0f;
						// std::cout << dist << std::endl;
					}
					else
					{	
						// If the ray has already intersected the volume and it is going out of boundaries
						if (intersected)
							break;
						// If not, conitnue traversing until finding intersection
						step += 1.0f;
						continue;
					}	 */								
				}	
			}
			
		}
	}

	writePointCloud("pointcloud.off", vertices);

	return 0;
}