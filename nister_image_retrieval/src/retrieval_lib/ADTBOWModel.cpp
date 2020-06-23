#include "ADTBOWmodel.hpp"


#include "ADTfilesystem.hpp"
#include "ADTVision.hpp"
#include <iostream>
#include <memory>
#include <fstream>


// --------------------------------------------------------------------------

BowVector::BowVector(void)
{
}

// --------------------------------------------------------------------------

BowVector::~BowVector(void)
{
}

// --------------------------------------------------------------------------

void BowVector::addWeight(uint32_t id, double v)
{
  BowVector::iterator vit = this->lower_bound(id);
  
  if(vit != this->end() && !(this->key_comp()(id, vit->first)))
  {
    vit->second += v;
  }
  else
  {
    this->insert(vit, BowVector::value_type(id, v));
  }
}

// --------------------------------------------------------------------------

void BowVector::addIfNotExist(uint32_t id, double v)
{
  BowVector::iterator vit = this->lower_bound(id);
  
  if(vit == this->end() || (this->key_comp()(id, vit->first)))
  {
    this->insert(vit, BowVector::value_type(id, v));
  }
}

// --------------------------------------------------------------------------

void BowVector::normalize(LNorm norm_type)
{
  double norm = 0.0; 
  BowVector::iterator it;

  if(norm_type == L1)
  {
    for(it = begin(); it != end(); ++it)
      norm += fabs(it->second);
  }
  else
  {
    for(it = begin(); it != end(); ++it)
      norm += it->second * it->second;
		norm = sqrt(norm);  
  }

  if(norm > 0.0)
  {
    for(it = begin(); it != end(); ++it)
      it->second /= norm;
  }
}


// If you change the type of WordValue, make sure you change also the
// epsilon value (this is needed by the KL method)
const double GeneralScoring::LOG_EPS = log(DBL_EPSILON); // FLT_EPSILON

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double L1Scoring::score(const BowVector &v1, const BowVector &v2) const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const double& vi = v1_it->second;
    const double& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += fabs(vi - wi) - fabs(vi) - fabs(wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
  //		for all i | v_i != 0 and w_i != 0 
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  score = -score/2.0;

  return score; // [0..1]
}
// --------------------------------------------------------------------------

std::ostream& operator<< (std::ostream &out, const BowVector &v)
{
  BowVector::const_iterator vit;
  std::vector<unsigned int>::const_iterator iit;
  unsigned int i = 0; 
  const unsigned int N = v.size();
  for(vit = v.begin(); vit != v.end(); ++vit, ++i)
  {
    out << "<" << vit->first << ", " << vit->second << ">";
    
    if(i < N-1) out << ", ";
  }
  return out;
}

// --------------------------------------------------------------------------

void BowVector::saveM(const std::string &filename, size_t W) const
{
  std::fstream f(filename.c_str(), std::ios::out);
  
  uint32_t last = 0;
  BowVector::const_iterator bit;
  for(bit = this->begin(); bit != this->end(); ++bit)
  {
    for(; last < bit->first; ++last)
    {
      f << "0 ";
    }
    f << bit->second << " ";
    
    last = bit->first + 1;
  }
  for(; last < (uint32_t)W; ++last)
    f << "0 ";
  
  f.close();
}


ADTBagofWords::ADTBagofWords(std::shared_ptr<TrainParams> bwPin): ADTSearchBase() 
{
	bwParams = bwPin;
}

/* */
void  ADTBagofWords::compute_bow_features(const std::shared_ptr<ADTDataset> &dataset)
{
	const std::vector<std::shared_ptr<const ADTImage> >	&all_images = dataset->all_images();
	const cv::Ptr<cv::DescriptorMatcher> &matcher = ADTVision::construct_descriptor_matcher(vocabulary_matrix);

	for(int64_t i = 0; i < (int64_t) all_images.size(); i++)
	{
		const std::string & sift_descriptor_location = dataset->feat_location(all_images[i]->feature_path("descriptors"));
		const std::string & bow_descriptor_location = dataset->feat_location(all_images[i]->feature_path("bow_descriptors"));
		cv::Mat descriptors, bow_descriptors, descriptorsf;

		if(!adtfilesystem::file_exists(sift_descriptor_location))
			continue;

		if(!adtfilesystem::load_cvmat(sift_descriptor_location, descriptors))
			continue;

		descriptors.convertTo(descriptorsf, CV_32FC1);
		adtfilesystem::create_file_directory(bow_descriptor_location);

		if(!ADTVision::compute_bow_feature(descriptorsf,matcher,bow_descriptors,std::shared_ptr < std::vector < std::vector<uint32_t>>>()))
			continue;

		const std::vector<std::pair<uint32_t, float> >	&bow_descriptors_sparse = adtmath::sparsify(bow_descriptors);
		adtfilesystem::write_sparse_vector(bow_descriptor_location, bow_descriptors_sparse);
		INFO( "Wrote " + bow_descriptor_location);
	}
}

ADTBagofWords::ADTBagofWords(const std::string &file_path) : ADTSearchBase(file_path) 
{
	if(!adtfilesystem::file_exists(file_path)) 
	{
		std::cerr << "Error reading bag of words from " << file_path << std::endl;
		return;
	}
	
	if(!this->load(file_path)) 
	{
		INFO("Error reading bag of words from " + file_path);
	}
}

bool ADTBagofWords::load (const std::string &file_path) 
{
	INFO("Reading bag of words from " + file_path + "...");
	if (!adtfilesystem::load_cvmat(file_path, vocabulary_matrix)) 
	{
		std::cerr << "Failed to read vocabulary from " << file_path << std::endl;
		return false;
	}

	INFO("Done reading bag of words.");
	
	return true;
}


bool ADTBagofWords::save (const std::string &file_path) const 
{
	INFO("Writing bag of words to " + file_path + "...");

	adtfilesystem::create_file_directory(file_path);
	if (!adtfilesystem::write_cvmat(file_path, vocabulary_matrix)) 
	{
		std::cerr << "Failed to write vocabulary to " << file_path << std::endl;
		return false;
	}

	INFO("Done writing bag of words.");
	return true;
}

bool ADTBagofWords::train(const std::shared_ptr<ADTDataset> &dataset) 
{
	uint32_t k = bwParams->numClusters;
	uint32_t n = bwParams->numFeatures;
	descType = bwParams->descType;
	int numbofTraining = bwParams->numberofTrainingImgs;
	std::vector<std::shared_ptr<const ADTImage> > random_training_images = dataset->random_images(numbofTraining);
	std::vector<uint64_t> all_ids(random_training_images.size());
	for (uint64_t i = 0; i < random_training_images.size(); i++) 
	{
		all_ids[i] = random_training_images[i]->id;
	}
	std::random_shuffle(all_ids.begin(), all_ids.end());

	std::vector<cv::Mat> all_descriptors;
	uint64_t num_features = 0;
	for (size_t i = 0; i < all_ids.size(); i++) 
	{
		std::shared_ptr<ADTImage> image = std::static_pointer_cast<ADTImage>(dataset->image(all_ids[i]));
		if (image == std::shared_ptr<ADTImage>()) 
			continue;

		const std::string &descriptors_location = dataset->feat_location(image->feature_path("descriptors"));
		if (!adtfilesystem::file_exists(descriptors_location)) 
			continue;

		cv::Mat descriptors, descriptorsf;
		if (adtfilesystem::load_cvmat(descriptors_location, descriptors)) 
		{
			num_features += descriptors.rows;
			if (n > 0 && num_features > n)
				break;
			descriptors.convertTo(descriptorsf, CV_32FC1);
			all_descriptors.push_back(descriptorsf);
		}
	}

	const cv::Mat merged_descriptor = ADTVision::merge_descriptors(all_descriptors, true);
	
	cv::Mat labels;
	uint32_t attempts = 1;
	cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 16, 0.0001);
	if (k > merged_descriptor.rows) 
	{ // k > n 
		std::cerr << "Warning: # clusters > # features, automatically setting #clusters = #features." << std::endl;
		k = merged_descriptor.rows;
	}
	
	cv::kmeans(merged_descriptor, k, labels, tc, attempts, cv::KMEANS_PP_CENTERS, vocabulary_matrix);

	return true;
}

std::shared_ptr<MatchResultsBase> ADTBagofWords::search( const std::shared_ptr<const SearchParamsBase> &params,const std::shared_ptr<ADTDataset> &datasetTest, const std::shared_ptr<const ADTImage > &example) 
{
	assert(0);
	return std::shared_ptr<MatchResultsBase>();
}

cv::Mat ADTBagofWords::vocabulary() const 
{
	return vocabulary_matrix;
}

uint32_t ADTBagofWords::num_clusters() const 
{
	return bwParams->numClusters;
}
