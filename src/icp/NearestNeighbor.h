#pragma once
//#include <nanoflann.hpp>
#include <flann/flann.hpp>

#include "Eigen.h"



class NearestNeighborSearch {
public:
	virtual ~NearestNeighborSearch() {}

	virtual void setMatchingMaxDistance(float maxDistance) {
		m_maxDistance = maxDistance;
	}

	virtual void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) = 0;
	virtual std::vector<int> queryMatches(const std::vector<Vector3f>& transformedPoints) = 0;

protected:
	float m_maxDistance;

	NearestNeighborSearch() : m_maxDistance{ 0.005f } {}
};


/**
 * Nearest neighbor search using FLANN.
 */
class NearestNeighborSearchFlann : public NearestNeighborSearch {
public:
	NearestNeighborSearchFlann() :
		NearestNeighborSearch(),
		m_nTrees{ 1 },
		m_index{ nullptr },
		m_flatPoints{ nullptr }
	{ }

	~NearestNeighborSearchFlann() {
		if (m_index) {
			delete m_flatPoints;
			delete m_index;
			m_flatPoints = nullptr;
			m_index = nullptr;
		}
	}

	void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) {

		// FLANN requires that all the points be flat. Therefore we copy the points to a separate flat array.
		m_flatPoints = new float[targetPoints.size() * 3];
		for (size_t pointIndex = 0; pointIndex < targetPoints.size(); pointIndex++) {
			for (size_t dim = 0; dim < 3; dim++) {
				m_flatPoints[pointIndex * 3 + dim] = targetPoints[pointIndex][dim];
			}
		}

		flann::Matrix<float> dataset(m_flatPoints, targetPoints.size(), 3);

		// Building the index takes some time.
		m_index = new flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(m_nTrees));
		m_index->buildIndex();

		std::cout << "FLANN index created." << std::endl;
	}

	std::vector<int> queryMatches(const std::vector<Vector3f>& transformedPoints) {
        if (!m_index) {
			std::cout << "FLANN index needs to be build before querying any matches." << std::endl;
			return {};
		}

		// FLANN requires that all the points be flat. Therefore we copy the points to a separate flat array.
		float* queryPoints = new float[transformedPoints.size() * 3];
		for (size_t pointIndex = 0; pointIndex < transformedPoints.size(); pointIndex++) {
			for (size_t dim = 0; dim < 3; dim++) {
				queryPoints[pointIndex * 3 + dim] = transformedPoints[pointIndex][dim];
			}
		}

		flann::Matrix<float> query(queryPoints, transformedPoints.size(), 3);
		flann::Matrix<int> indices(new int[query.rows * 1], query.rows, 1);
		flann::Matrix<float> distances(new float[query.rows * 1], query.rows, 1);

        //for (size_t i = 0; i < transformedPoints.size(); i++)
        //{
        //    std::cout << "Query: " << *query[i] << std::endl;
        //}
		
		// Do a knn search, searching for 1 nearest point and using 16 checks.
		flann::SearchParams searchParams{ 16 };
		searchParams.cores = 0;
		m_index->knnSearch(query, indices, distances, 1, searchParams);

		// Filter the matches.
		const unsigned nMatches = transformedPoints.size();
		std::vector<int> matches;
		matches.reserve(nMatches);

		for (int i = 0; i < nMatches; ++i) {
			if (*distances[i] <= m_maxDistance)
				matches.push_back( *indices[i]);
			else
				matches.push_back( -1);
		}
		
		// Release the memory.
		delete[] query.ptr();
		delete[] indices.ptr();
		delete[] distances.ptr();
		
		return matches;
	}

private:
	int m_nTrees;
	flann::Index<flann::L2<float>>* m_index;
	float* m_flatPoints;
};


