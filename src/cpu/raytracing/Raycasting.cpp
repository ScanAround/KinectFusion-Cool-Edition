#include "Raycasting.h"
#include "Ray.h"


/*Raycasting::Raycasting(const Volume& _tsdf, const Eigen::Matrix3f& _extrinsics, const Eigen::Vector3f _cameraCenter): 
tsdf(_tsdf), extrinsincs(_extrinsics), cameraCenter(_cameraCenter)
{
    width = 640;
    height = 480;
    intrinsics <<   525.0f, 0.0f, 319.5f,
                    0.0f, 525.0f, 239.5f,
                    0.0f, 0.0f, 1.0f;
}*/

/*Raycasting::Raycasting(kinect_fusion::VoxelGrid& _tsdf, const Eigen::Matrix3f& _extrinsics, const Eigen::Vector3f _cameraCenter):
tsdf(_tsdf), extrinsincs(_extrinsics), cameraCenter(_cameraCenter)
{
    width = 640;
    height = 480;
    intrinsics <<   525.0f, 0.0f, 319.5f,
                    0.0f, 525.0f, 239.5f,
                    0.0f, 0.0f, 1.0f;
}*/

Vertex Raycasting::castOne(const unsigned int i, const unsigned int j)
{
    Eigen::Vector3f rayOrigin = tsdf.worldToGrid(cameraCenter);
    Eigen::Vector3f rayNext(float(i), float(j), 1.0f);
    Eigen::Vector3f rayNextCameraSpace = intrinsics.inverse() * rayNext;
    Eigen::Vector3f rayNextWorldSpace = extrinsincs * rayNextCameraSpace + cameraCenter;
    Eigen::Vector3f rayNextGridSpace = tsdf.worldToGrid(rayNextWorldSpace);

    Eigen::Vector3f rayDir = rayNextGridSpace - rayOrigin;
    rayDir.normalize();

    Ray r(rayOrigin, rayDir);

    // TODO: calculate first intersection with the volume (if exists)
    // First, let's try step size equal to one single voxel
    float step = 1.0f;
    double prevDist = 0;
    bool intersected = false;

    for (unsigned int s = 0; s < MAX_MARCHING_STEPS; ++s)
    {
        Eigen::Vector3f p = r.at(step);
        
        // Think carefully if this cast is correct or not
        if (!tsdf.outOfVolume(int(p[0]), int(p[1]), int(p[2])))
        {
            intersected = true;
            double dist = tsdf.get(p.cast<int>());
            if (prevDist > 0 && dist <= 0 && s > 0)
            {	
                Eigen::Vector3f n = computeNormal(p);
                // Eigen::Vector3f interpolatedP = getInterpolatedIntersection(vol, rayOrigin, rayDir, step);
                Vertex v = {
                    p,  // position
                    n // normal
                };

                return v;
            }
            prevDist = dist;
            // step += (dist / tsdf.ddx) * 0.5f;
            step += 1.0f;
        }
        else
        {	
            // If the ray has already intersected the volume and it is going out of boundaries
            if (intersected)
            {
                Eigen::Vector3f p(MINF, MINF, MINF);
                Eigen::Vector3f n(MINF, MINF, MINF);
                Vertex v = {
                    p,  // position
                    n // normal
                };

                return v;
            }
            // If not, conitnue traversing until finding intersection
            step += 1.0f;
            continue;
        }						
    }

    Eigen::Vector3f p(MINF, MINF, MINF);
    Eigen::Vector3f n(MINF, MINF, MINF);
    Vertex v = {
        p,  // position
        n // normal
    };

    return v;		
}


void Raycasting::castAll()
{
    for (unsigned int j = 0; j < height; j += 6)  // CHANGE TO 1
	{
		for (unsigned int i = 0; i < width; i += 6)
		{
            Vertex v = Raycasting::castOne(i, j);
            if (v.normal[0] == MINF && v.normal[1] == MINF && v.normal[2] == MINF)
                continue;
            vertices.push_back(v);
        }
    }
}


Eigen::Vector3f Raycasting::computeNormal(const Eigen::Vector3f& p)
{
	Eigen::Vector3i pInt = p.cast<int>();
	// Numerical derivatives
	if (!tsdf.outOfVolume(pInt[0] + 1, pInt[1], pInt[2]) && 
		!tsdf.outOfVolume(pInt[0] - 1, pInt[1], pInt[2]) && 
		!tsdf.outOfVolume(pInt[0], pInt[1] + 1, pInt[2]) && 
		!tsdf.outOfVolume(pInt[0], pInt[1] - 1, pInt[2]) && 
		!tsdf.outOfVolume(pInt[0], pInt[1], pInt[2] + 1) && 
		!tsdf.outOfVolume(pInt[0], pInt[1], pInt[2] -1))
	{
		double deltaX = tsdf.get(pInt[0] + 1, pInt[1], pInt[2]) - tsdf.get(pInt[0] - 1, pInt[1], pInt[2]);
		double deltaY = tsdf.get(pInt[0], pInt[1] + 1, pInt[2]) - tsdf.get(pInt[0], pInt[1] - 1, pInt[2]);
		double deltaZ = tsdf.get(pInt[0], pInt[1], pInt[2] + 1) - tsdf.get(pInt[0], pInt[1], pInt[2] - 1);
		
		double gradX = deltaX / 2.0f;
		double gradY = deltaY / 2.0f;
		double gradZ = deltaZ / 2.0f;

		Eigen::Vector3f normal(gradX, gradY, gradZ);
		normal.normalize();
        return normal;
	}
	else
    {
         Eigen::Vector3f normal(MINF, MINF, MINF);
         return normal;
    }
}

/*
std::vector<Eigen::Vector3f> Raycasting::getVertices()
{
    std::vector<Eigen::Vector3f> vrtxs;
    for (Vertex& v : vertices)
    {
        Eigen::Vector3f posWorld = tsdf.gridToWorld(int(v.position[0]), int(v.position[1]), int(v.position[2]));
        vrtxs.push_back(posWorld);
    }
    return vrtxs;
}

std::vector<Eigen::Vector3f> Raycasting::getNormals()
{
    std::vector<Eigen::Vector3f> nrmls;

    for (Vertex& v : vertices)
        nrmls.push_back(v.normal);

    return nrmls;
}*/
