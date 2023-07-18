#pragma once
#include <flann/flann.h>
#include <eigen3/Eigen/Dense>


//adapted from exercise 5
class NN_flann{
public:
    NN_flann(int m_nTrees, float m_maxDistance):
        m_nTrees{m_nTrees},
        m_index{nullptr},
        m_flatPoints{nullptr},
        m_maxDistance{m_maxDistance}{}
    
    ~NN_flann() {
        if(m_index){
            delete m_flatPoints;
			delete m_index;
			m_flatPoints = nullptr;
			m_index = nullptr;
        }
    }

    void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) {
		std::cout << "Initializing FLANN index with " << targetPoints.size() << " points." << std::endl;

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

	std::vector<std::pair<int, int>>  queryMatches(const std::vector<Eigen::Vector3f>& transformedPoints) {
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

		// Do a knn search, searching for 1 nearest point and using 16 checks.
		flann::SearchParams searchParams{ 16 };
		searchParams.cores = 0;
		m_index->knnSearch(query, indices, distances, 1, searchParams);

		// Filter the matches.
		const unsigned nMatches = transformedPoints.size();
		std::vector<std::pair<int, int>> matches;
		matches.reserve(nMatches);

		for (int i = 0; i < nMatches; ++i) {
			if (*distances[i] <= m_maxDistance)
				// matches.push_back(Match{ *indices[i], 1.f });
				matches.push_back(std::make_pair(i, *indices[i]));
		}

		// Release the memory.
		delete[] query.ptr();
		delete[] indices.ptr();
		delete[] distances.ptr();

		return matches;
	}

private:
    int m_nTrees;
    flann::Index<flann::L2<float>> * m_index;
    float* m_flatPoints; 
    float m_maxDistance; 
};