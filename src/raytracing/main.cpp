#define _USE_MATH_DEFINES  // to access M_PI macro from math.h

#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include "ImplicitSurface.h"
#include "Volume.h"

// TODO: choose optimal truncation value
#define TRUNCATION 1.0
// #define MAX_MARCHING_STEPS 10000
#define MAX_MARCHING_STEPS 500
#define EPSILON 0.1


struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0) -> why?
	Eigen::Vector3f position;
	Eigen::Vector3f normal;
};

void writePointCloud(const std::string& filename, const std::vector<Vertex>& _vertices, bool includeNormals=false)
{
	std::ofstream file(filename);
	file << "OFF" << std::endl;
	file << _vertices.size() << " 0 0" << std::endl;
	for (unsigned int i = 0; i < _vertices.size(); ++i)
	{
		file << _vertices[i].position[0] << " " << _vertices[i].position[1] << " " << _vertices[i].position[2];
		if (includeNormals)
			file << " " << _vertices[i].normal[0] << " " << _vertices[i].normal[1] << " " << _vertices[i].normal[2] << std::endl;
		else
			file << std::endl;
	}
}
float trilinearInterpolation(const Eigen::Vector3f& point,
	Volume& volume,
	const int voxel_grid_dim_x,
	const int voxel_grid_dim_y,
	const int voxel_grid_dim_z) {

	Eigen::Vector3i point_in_grid = point.cast<int>();

	const float vx = (static_cast<float>(point_in_grid[0]) + 0.5f);
	const float vy = (static_cast<float>(point_in_grid[1]) + 0.5f);
	const float vz = (static_cast<float>(point_in_grid[2]) + 0.5f);

	point_in_grid[0] = (point[0] < vx) ? (point_in_grid[0] - 1) : point_in_grid[0];
	point_in_grid[1] = (point[1] < vy) ? (point_in_grid[1] - 1) : point_in_grid[1];
	point_in_grid[2] = (point[2] < vz) ? (point_in_grid[2] - 1) : point_in_grid[2];

	const float a = (point.x() - (static_cast<float>(point_in_grid[0]) + 0.5f));
	const float b = (point.y() - (static_cast<float>(point_in_grid[0]) + 0.5f));
	const float c = (point.z() - (static_cast<float>(point_in_grid[0]) + 0.5f));

	const int xd = point_in_grid[0];
	const int yd = point_in_grid[1];
	const int zd = point_in_grid[2];
	//std::cout << "Volume" << volume;
	std::cout << "X,y,z Values:" << xd << " " << yd << " " << zd << "\n";
	std::cout << "Volume values: " << volume.getDimX() << " " << volume.getDimY() << " " << volume.getDimZ() << "\n ";
	std::cout << "Voxel_grid values: " << voxel_grid_dim_x << " " << voxel_grid_dim_x << " " << voxel_grid_dim_x << "\n ";
	const float c000 = volume.get((xd), (yd)*voxel_grid_dim_x, (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c000 << "\n";
	const float c001 = volume.get((xd), (yd)*voxel_grid_dim_x, (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c001 << "\n";
	const float c010 = volume.get((xd), (yd + 1) * voxel_grid_dim_x, (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c010 << "\n";
	const float c011 = volume.get((xd), (yd + 1) * voxel_grid_dim_x, (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c011 << "\n";
	const float c100 = volume.get((xd + 1), (yd)*voxel_grid_dim_x, (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c100 << "\n";
	const float c101 = volume.get((xd + 1), (yd)*voxel_grid_dim_x, (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c101 << "\n";
	const float c110 = volume.get((xd + 1), (yd + 1) * voxel_grid_dim_x, (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c110 << "\n";
	const float c111 = volume.get((xd + 1), (yd + 1) * voxel_grid_dim_x, (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c111 << "\n";
	return c000 * (1 - a) * (1 - b) * (1 - c) +
		c001 * (1 - a) * (1 - b) * c +
		c010 * (1 - a) * b * (1 - c) +
		c011 * (1 - a) * b * c +
		c100 * a * (1 - b) * (1 - c) +
		c101 * a * (1 - b) * c +
		c110 * a * b * (1 - c) +
		c111 * a * b * c;
}

Eigen::Vector3f getNormal(Volume& vol, const Eigen::Vector3f& p)
{
	Eigen::Vector3i pInt = p.cast<int>();
	// Numerical derivatives
	if (!vol.outOfVolume(pInt[0] + 1, pInt[1], pInt[2]) && 
		!vol.outOfVolume(pInt[0] - 1, pInt[1], pInt[2]) && 
		!vol.outOfVolume(pInt[0], pInt[1] + 1, pInt[2]) && 
		!vol.outOfVolume(pInt[0], pInt[1] - 1, pInt[2]) && 
		!vol.outOfVolume(pInt[0], pInt[1], pInt[2] + 1) && 
		!vol.outOfVolume(pInt[0], pInt[1], pInt[2] -1))
	{
		double deltaX = vol.get(pInt[0] + 1, pInt[1], pInt[2]) - vol.get(pInt[0] - 1, pInt[1], pInt[2]);
		double deltaY = vol.get(pInt[0], pInt[1] + 1, pInt[2]) - vol.get(pInt[0], pInt[1] - 1, pInt[2]);
		double deltaZ = vol.get(pInt[0], pInt[1], pInt[2] + 1) - vol.get(pInt[0], pInt[1], pInt[2] - 1);
		
		double gradX = deltaX / 2.0f;
		double gradY = deltaY / 2.0f;
		double gradZ = deltaZ / 2.0f;

		Eigen::Vector3f normal(gradX, gradY, gradZ);
		normal.normalize();

		return normal;
	}
	// TODO: change to M_INF
	Eigen::Vector3f normal(0.0f, 0.0f, 0.0f);
	
}

Eigen::Vector3f getInterpolatedIntersection(Volume& vol, const Eigen::Vector3f& origin, const Eigen::Vector3f& dir, double step)
{
	Eigen::Vector3f t = origin + (step - 1) * dir; // before crossing
	Eigen::Vector3f delta = origin + step * dir;  // after crossing
	double tValue = vol.get(t.cast<int>());
	double deltaValue = vol.get(delta.cast<int>());
	// Now this is trivial because step size is 1
	// TODO: step in grid CS is correct? Or should it be WCS?
	double deltaT = step - (step - 1);
	double interpolatedT = (step - 1) - (deltaT * tValue / (deltaValue - tValue));

	Eigen::Vector3f interpolatedV = origin + interpolatedT * dir;

	return interpolatedV;
}

int main()
{
	const auto imageWidth = 640; 
	const auto imageHeight = 480;

	Eigen::Matrix3f intrinsics; 
	intrinsics <<   525.0f, 0.0f, 319.5f,
					0.0f, 525.0f, 239.5f,
					0.0f, 0.0f, 1.0f;

	// Eigen::Vector3f cameraCenter(2.0f, 2.0f, -2.0f);
	Eigen::Vector3f cameraCenter(1.5f, 1.5f, -1.5f);

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

	std::vector<Vertex> vertices,normals_vertex;
	std::vector<Eigen::Vector3f> normals;
	/* Vertex c = {
		cameraCenter
	};
	vertices.push_back(c); */

	// Init implicit surface
	// Torus implicitTorus = Torus(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4, 0.1);
	Sphere implicit = Sphere(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4);
	// Fill spatial grid with distance to the implicit surface
	// unsigned int mc_res = 600;
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

				/* if ( x % 10 == 0 && y % 10 == 0 && z % 10 == 0)
				{
					Vertex v = {
							p.cast<float>()  // position
						};
					vertices.push_back(v);
				} */
					
			}
		}
	}

	// Test function to check the point cloud writer
	// vol.writePointCloud("pointcloud.off");

	Eigen::Vector3f rayOrigin = vol.worldToGrid(cameraCenter);

	// Traverse the image pixel by pixel
	for (unsigned int j = 0; j < imageHeight; j += 4)  // CHANGE TO 1
	{
		for (unsigned int i = 0; i < imageWidth; i += 4)
		{
			Eigen::Vector3f rayNext(float(i), float(j), 1.0f);
			Eigen::Vector3f rayNextCameraSpace = intrinsics.inverse() * rayNext;
			Eigen::Vector3f rayNextWorldSpace = rotation * rayNextCameraSpace + cameraCenter;
			Eigen::Vector3f rayNextGridSpace = vol.worldToGrid(rayNextWorldSpace);

			Eigen::Vector3f rayDir = rayNextGridSpace - rayOrigin;
			//Eigen::Vector3f rayDir = rayNextWorldSpace - cameraCenter;
			rayDir.normalize();

			// TODO: calculate first intersection with the volume (if exists)
			// First, let's try step size equal to one single voxel
			float step = 1.0f;
			double prevDist = 0;
			bool intersected = false;
			/* if (j % 10 == 0 && i % 10 == 0)
			{ */
			for (unsigned int s = 0; s < MAX_MARCHING_STEPS; ++s)
			{
				Eigen::Vector3f p = rayOrigin + step * rayDir;
				/* Vertex v = {
							p  // position
						};
				vertices.push_back(v);
				step += .5f; */
				
				// Think carefully if this cast is correct or not
				if (!vol.outOfVolume(int(p[0]), int(p[1]), int(p[2])))
				{
					intersected = true;
					double dist = vol.get(p.cast<int>());
					if (prevDist > 0 && dist <=0 && s > 0)
					{	
						// Eigen::Vector3f n = getNormal(vol, p);
						// If normal is not a valid vector
						// if (n[0] == 0.0f && n[1] == 0.0f && n[2] == 0.0f)
						// 	break;
						float t =trilinearInterpolation(p, vol, 1, 1, 1);
						std::cout << t << " \n";
						Eigen::Vector3f interpolatedP = getInterpolatedIntersection(vol, rayOrigin, rayDir, step);
						Vertex v = {
							interpolatedP,  // position
						 	// n  // normal
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
				}						
			}	
			//}
			
		}
	}

	writePointCloud("C:/Users/yigitavci/Desktop/TUM_DERS/Semester_2/3D_Scanning/KinectFusion-Cool-Edition/src/raytracing/pointcloud2.off", vertices, false);

	return 0;
}