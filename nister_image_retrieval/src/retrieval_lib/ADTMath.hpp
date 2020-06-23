#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Simple class for specific math data operators such as sparsify*/
/*etc.*/

#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/random.hpp>

namespace adtmath 
{	
	typedef std::vector< std::pair<uint32_t, float > > sparse_vector_t;
	std::vector< std::pair<uint32_t, float> > sparsify(const cv::Mat &dense);
	std::vector<float> dense1d(const cv::Mat &dense);

	/// Converts the cosine similarity between two sparse weight vectors, which are premultiplied
	/// by the relevant entries in idfw.  Sample usage would be weights0 and weights1 to represent
	/// two BoW vectors, and idfw to represent a vector of inverse document frequencies.
	float cos_sim(const std::vector<std::pair<uint32_t, float> > &weights0, const std::vector<std::pair<uint32_t, float> > &weights1,const std::vector<float> &idfw);

	/// Converts the histogram intersection (min) between two sparse weight vectors, which are premultiplied
	/// by the relevant entries in idfw.  Sample usage would be weights0 and weights1 to represent
	/// two BoW vectors, and idfw to represent a vector of inverse document frequencies.
	float min_hist(const std::vector<std::pair<uint32_t, float> > &weights0,const std::vector<std::pair<uint32_t, float> > &weights1,const std::vector<float> &idfw);
}
