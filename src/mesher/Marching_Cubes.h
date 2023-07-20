#pragma once

#include <eigen3/Eigen/Dense>
#include "../tsdf/voxel_grid.h"
#include "Tables.h"

struct MC_Grid{
    Voxel v[8];
}

class Marching_Cubes{
    public:
        static void Mesher(kinect_fusion::VoxelGrid& grid, std::string obj_file){
            // we want to create a mesh from an input voxel grid

            for(int x = 0; x < grid.getDimX() - 1; ++x){
                for(int y = 0; y < grid.getDimY() - 1; ++y){
                    for(int z = 0; z < grid.getDimZ() - 1; ++z){
                        MC_Grid cell;
                        cell.v[0] = grid.getVoxel(x, y, z); 
                        cell.v[1] = grid.getVoxel(x-1, y, z); 
                        cell.v[2] = grid.getVoxel(x, y-1, z); 
                        cell.v[3] = grid.getVoxel(x, y, z-1); 
                        cell.v[4] = grid.getVoxel(x-1, y-1, z); 
                        cell.v[5] = grid.getVoxel(x, y-1, z-1); 
                        cell.v[6] = grid.getVoxel(x-1, y, z-1); 
                        cell.v[7] = grid.getVoxel(x-1, y-1, z-1); 

                    }
                }
            }
        };

        int Polygonise(MC_Gridcell grid, double isolevel, MC_Triangle* triangles) {
            // taken from 3D scanning and Motion capture tutorial!
            // checks if voxels are below isolevel and if so uses the interpolate function to find out where the vertex is and which triangle it would return
        
            int ntriang;
            int cubeindex;
            Vector3d vertlist[12];

            cubeindex = 0; // gets the unique cubeindex based on if each corner is less than the iso level
            if (grid.val[0] < isolevel) cubeindex |= 1;
            if (grid.val[1] < isolevel) cubeindex |= 2;
            if (grid.val[2] < isolevel) cubeindex |= 4;
            if (grid.val[3] < isolevel) cubeindex |= 8;
            if (grid.val[4] < isolevel) cubeindex |= 16;
            if (grid.val[5] < isolevel) cubeindex |= 32;
            if (grid.val[6] < isolevel) cubeindex |= 64;
            if (grid.val[7] < isolevel) cubeindex |= 128;

            /* Cube is entirely in/out of the surface */
            if (edgeTable[cubeindex] == 0)
                return 0;

            /* Find the vertices where the surface intersects the cube */
            if (edgeTable[cubeindex] & 1)
                vertlist[0] = interpolate(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
            if (edgeTable[cubeindex] & 2)
                vertlist[1] = interpolate(isolevel, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
            if (edgeTable[cubeindex] & 4)
                vertlist[2] = interpolate(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
            if (edgeTable[cubeindex] & 8)
                vertlist[3] = interpolate(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
            if (edgeTable[cubeindex] & 16)
                vertlist[4] = interpolate(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
            if (edgeTable[cubeindex] & 32)
                vertlist[5] = interpolate(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
            if (edgeTable[cubeindex] & 64)
                vertlist[6] = interpolate(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
            if (edgeTable[cubeindex] & 128)
                vertlist[7] = interpolate(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
            if (edgeTable[cubeindex] & 256)
                vertlist[8] = interpolate(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
            if (edgeTable[cubeindex] & 512)
                vertlist[9] = interpolate(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
            if (edgeTable[cubeindex] & 1024)
                vertlist[10] = interpolate(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
            if (edgeTable[cubeindex] & 2048)
                vertlist[11] = interpolate(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

            /* Create the triangle */
            ntriang = 0;
            for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
                triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i]];
                triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i + 1]];
                triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i + 2]];
                ntriang++;
            }

            return ntriang;
        }
        
        inline Eigen::Vector3d interpolate(Voxel& voxel1, Voxel& voxel2){
            // finding the zero crossing value p between voxel1 and voxel2
            double lambda = voxel1.tsdfValue / (voxel1.tsdfValue - voxel2.tsdfValue)
            return (voxel2.position - voxel1.position) * lambda + p1;
        };

        static void Mesher(std::string grid_file, std::string obj_file){
            
        };
    private:
};