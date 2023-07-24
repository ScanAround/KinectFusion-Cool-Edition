#ifndef RAYCASTING_H
#define RAYCASTING_H

#include <vector>
#include "Ray.h"

class Raycasting
{
public:
	Raycasting(): a(0.0)
	{

	}

	std::vector<Vertex> castAll();

	Vertex castOne(const Ray& r);

private:
	double a;
};


#endif // !RAYCASTING_H