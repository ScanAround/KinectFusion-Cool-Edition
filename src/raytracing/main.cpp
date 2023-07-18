#define _USE_MATH_DEFINES  // to access M_PI macro from math.h

#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include "ImplicitSurface.h"
#include "Volume.h"

// TODO: choose optimal truncation value
#define TRUNCATION 1.0
#define MAX_MARCHING_STEPS 10000
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
float trilinearInterpolation(const Eigen::Vector3f& point,
	Volume volume,
	const int voxel_grid_dim_x,
	const int voxel_grid_dim_y,
	const int voxel_grid_dim_z) {

	Eigen::Vector3i point_in_grid = point.cast<int>();

	const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
	const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
	const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

	point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
	point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
	point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

	const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
	const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
	const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

	const int xd = point_in_grid.x();
	const int yd = point_in_grid.y();
	const int zd = point_in_grid.z();
	//std::cout << "Volume" << volume;
	std::cout << "X,y,z Values:" << xd << " " << yd << " " << zd << "\n";
	std::cout << "Volume values: " << volume.getDimX() << " " << volume.getDimY() << " " << volume.getDimZ() << "\n ";
	std::cout << "Voxel_grid values: " << voxel_grid_dim_x << " " << voxel_grid_dim_x << " " << voxel_grid_dim_x << "\n ";
	const float c000 = volume.get((xd),(yd)*voxel_grid_dim_x ,(zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c000 << "\n";
	const float c001 = volume.get((xd),(yd)*voxel_grid_dim_x , (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c001 << "\n";
	const float c010 = volume.get((xd),(yd + 1) * voxel_grid_dim_x , (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c010 << "\n";
	const float c011 = volume.get((xd),(yd + 1) * voxel_grid_dim_x , (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c011 << "\n";
	const float c100 = volume.get((xd + 1) , (yd)*voxel_grid_dim_x , (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c100 << "\n";
	const float c101 = volume.get((xd + 1) , (yd)*voxel_grid_dim_x , (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c101 << "\n";
	const float c110 = volume.get((xd + 1) , (yd + 1) * voxel_grid_dim_x , (zd)*voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c110 << "\n";
	const float c111 = volume.get((xd + 1) , (yd + 1) * voxel_grid_dim_x , (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y);
	std::cout << c111 << "\n";

	std::cout << "test_interp";
	return c000 * (1 - a) * (1 - b) * (1 - c) +
		c001 * (1 - a) * (1 - b) * c +
		c010 * (1 - a) * b * (1 - c) +
		c011 * (1 - a) * b * c +
		c100 * a * (1 - b) * (1 - c) +
		c101 * a * (1 - b) * c +
		c110 * a * b * (1 - c) +
		c111 * a * b * c;
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
	unsigned int mc_res = 600;
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
	for (unsigned int j = 0; j < imageHeight; ++j)
	{
		for (unsigned int i = 0; i < imageWidth; ++i)
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
						Vertex v = {
							p  // position
						};
						vertices.push_back(v);

						// I could not fully understand why we do this
						const Eigen::Vector3f location_in_grid = p-cameraCenter;
						Eigen::Vector3f normal, shifted;


						// Dimensions of the volume
						int voxel_grid_dim_x = 1;
						int voxel_grid_dim_y = 1;
						int voxel_grid_dim_z = 1;
						// X direction normal calculation
						shifted = location_in_grid;
						shifted.x() += 1;
						if (shifted.x() >= vol.getDimX() - 1) {
							break;
						}
						
						const float Fx1 = trilinearInterpolation(shifted, vol, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

						shifted = location_in_grid;
						shifted.x() -= 1;
						if (shifted.x() < 1) {
							break;
						}
						const float Fx2 = trilinearInterpolation(shifted, vol, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
						std::cout << "testt";
						normal.x() = (Fx1 - Fx2);
						std::cout << "test_end";
						// Y direction normal calculation
						shifted = location_in_grid;
						shifted.y() += 1;
						if (shifted.y() >= vol.getDimY() - 1) {
							break;
						}
						const float Fy1 = trilinearInterpolation(shifted, vol, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

						shifted = location_in_grid;
						shifted.y() -= 1;
						if (shifted.y() < 1) {
							break;
						}
						const float Fy2 = trilinearInterpolation(shifted, vol, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

						normal.y() = (Fy1 - Fy2);

						// Z direction normal calculation
						shifted = location_in_grid;
						shifted.z() += 1;
						if (shifted.z() >= vol.getDimZ() - 1) {
							break;
						}
						const float Fz1 = trilinearInterpolation(shifted, vol, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

						shifted = location_in_grid;
						shifted.z() -= 1;
						if (shifted.z() < 1) {
							break;
						}
						const float Fz2 = trilinearInterpolation(shifted, vol, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

						normal.z() = (Fz1 - Fz2);

						if (normal.norm() == 0) {
							break;
						}
						Vertex vv = {
							normal  // position
						};
						//std::cout << normal << " ";
						normals_vertex.push_back(vv);
						normal.normalize();
						normals.push_back(normal);

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

	writePointCloud("C:/Users/yigitavci/Desktop/TUM_DERS/Semester_2/3D_Scanning/KinectFusion-Cool-Edition/src/raytracing/pointcloud.off", vertices);
	writePointCloud("C:/Users/yigitavci/Desktop/TUM_DERS/Semester_2/3D_Scanning/KinectFusion-Cool-Edition/src/raytracing/pointcloud_normals.off", normals_vertex);


	return 0;
}