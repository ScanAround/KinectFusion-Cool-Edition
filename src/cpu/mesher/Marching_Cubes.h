#pragma once

#include <Eigen/Dense>
#include "../tsdf/voxel_grid.h"
#include "../tsdf/voxel.h"
#include "Lookup_Tables.h"
#include <fstream>
#include <iostream>

class Marching_Cubes{
    public:
        struct MC_Grid{
            kinect_fusion::Voxel v[8];
        };

        struct MC_Triangle{
            Eigen::Vector3d p[3];
        };


        static void Mesher(kinect_fusion::VoxelGrid& grid, double isolevel, std::string off_dir = "mesh.off"){
            // we want to create a mesh from an input voxel grid
            std::cout << "creating mesh from tsdf" << std::endl;
            std::ofstream OffFile(off_dir);
            OffFile << "OFF" << std::endl;
            
            int tmp_num_faces, num_vertices = 0, num_faces = 0;

            std::vector<std::vector<double>> vertices;
            std::vector<std::vector<int>> faces;

            for(int x = 0; x < grid.getDimX() - 1; ++x){
                for(int y = 0; y < grid.getDimY() - 1; ++y){
                    for(int z = 0; z < grid.getDimZ() - 1; ++z){
                        MC_Grid cell;
                        MC_Triangle triangles[6];
                        
                        cell.v[0] = grid.getVoxel(x+1, y, z); 
                        cell.v[1] = grid.getVoxel(x, y, z); 
                        cell.v[2] = grid.getVoxel(x, y+1, z); 
                        cell.v[3] = grid.getVoxel(x+1, y+1, z); 
                        cell.v[4] = grid.getVoxel(x+1, y, z+1); 
                        cell.v[5] = grid.getVoxel(x, y, z+1); 
                        cell.v[6] = grid.getVoxel(x, y+1, z+1); 
                        cell.v[7] = grid.getVoxel(x+1, y+1, z+1); 

                        tmp_num_faces = Polyganize(cell, triangles, isolevel);

                        for(int triangle = 0; triangle < tmp_num_faces; ++triangle){
                            vertices.push_back({triangles[triangle].p[0][0], triangles[triangle].p[0][1],triangles[triangle].p[0][2]});
                            vertices.push_back({triangles[triangle].p[1][0], triangles[triangle].p[1][1],triangles[triangle].p[1][2]});
                            vertices.push_back({triangles[triangle].p[2][0], triangles[triangle].p[2][1],triangles[triangle].p[2][2]});
                            faces.push_back({num_vertices, num_vertices+1, num_vertices+2});
                            num_vertices += 3;
                            num_faces += 1;
                        }
                    }
                }
            }
            
            OffFile << num_vertices << " " << num_faces << " " << 0 << std::endl; 

            for(auto vertix: vertices){
                OffFile << vertix[0] << " " << vertix[1] << " " << vertix[2] << std::endl; 
            }
            for(auto face: faces){
                OffFile << 3 << " " << face[0] << " " << face[1] << " " << face[2] << std::endl; 
            }   
            std::cout << "meshfile has been saved" << std::endl;
            OffFile.close();
        };

        static int Polyganize(MC_Grid grid, MC_Triangle* triangles, double isolevel) {
            // adapted from 3D scanning and Motion capture tutorial
            // checks if voxels are below 0 and if so uses the interpolate function to find out where the vertex is and which triangle it would return

            int ntriang;
            int cubeindex;
            Eigen::Vector3d vertlist[12];

            cubeindex = 0; // gets the unique cubeindex based on if each corner is less than the iso level
            if (grid.v[0].tsdfValue < 0.0) cubeindex |= 1;
            if (grid.v[1].tsdfValue < 0.0) cubeindex |= 2;
            if (grid.v[2].tsdfValue < 0.0) cubeindex |= 4;
            if (grid.v[3].tsdfValue < 0.0) cubeindex |= 8;
            if (grid.v[4].tsdfValue < 0.0) cubeindex |= 16;
            if (grid.v[5].tsdfValue < 0.0) cubeindex |= 32;
            if (grid.v[6].tsdfValue < 0.0) cubeindex |= 64;
            if (grid.v[7].tsdfValue < 0.0) cubeindex |= 128;

            /* Cube is entirely in/out of the surface */
            if (edgeTable[cubeindex] == 0)
                return 0;

            /* Find the vertices where the surface intersects the cube */
            if (edgeTable[cubeindex] & 1)
                vertlist[0] = interpolate(grid.v[0], grid.v[1], isolevel);
            if (edgeTable[cubeindex] & 2)
                vertlist[1] = interpolate(grid.v[1], grid.v[2], isolevel);
            if (edgeTable[cubeindex] & 4)
                vertlist[2] = interpolate(grid.v[2], grid.v[3], isolevel);
            if (edgeTable[cubeindex] & 8)
                vertlist[3] = interpolate(grid.v[3], grid.v[0], isolevel);
            if (edgeTable[cubeindex] & 16)
                vertlist[4] = interpolate(grid.v[4], grid.v[5], isolevel);
            if (edgeTable[cubeindex] & 32)
                vertlist[5] = interpolate(grid.v[5], grid.v[6], isolevel);
            if (edgeTable[cubeindex] & 64)
                vertlist[6] = interpolate(grid.v[6], grid.v[7], isolevel);
            if (edgeTable[cubeindex] & 128)
                vertlist[7] = interpolate(grid.v[7], grid.v[4], isolevel);
            if (edgeTable[cubeindex] & 256)
                vertlist[8] = interpolate(grid.v[0], grid.v[4], isolevel);
            if (edgeTable[cubeindex] & 512)
                vertlist[9] = interpolate(grid.v[1], grid.v[5], isolevel);
            if (edgeTable[cubeindex] & 1024)
                vertlist[10] = interpolate(grid.v[2], grid.v[6], isolevel);
            if (edgeTable[cubeindex] & 2048)
                vertlist[11] = interpolate(grid.v[3], grid.v[7], isolevel);

            /* Create the triangle */
            ntriang = 0;
            for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
                if(!std::isnan(vertlist[triTable[cubeindex][i]][0])
                && !std::isnan(vertlist[triTable[cubeindex][i + 1]][0])
                && !std::isnan(vertlist[triTable[cubeindex][i + 2]][0])){
                    triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i]];
                    triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i + 1]];
                    triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i + 2]];
                    ntriang++;
                }
            }

            return ntriang;
        }

        static inline Eigen::Vector3d interpolate(kinect_fusion::Voxel & voxel1, kinect_fusion::Voxel& voxel2, double isolevel){
        // finding the zero crossing value p between voxel1 and voxel2
        double lambda = (voxel1.tsdfValue - isolevel) / (voxel1.tsdfValue - voxel2.tsdfValue);
        return (voxel2.position - voxel1.position) * lambda + voxel1.position;
    };
};