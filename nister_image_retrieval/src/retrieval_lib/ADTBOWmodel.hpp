#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Simple class which implements bag of words based image search. This can be used*/
/*to create bag of words model for a given vocabulary which can then be used for creating.*/
/*retrieval system by using inverted index*/

#include "ADTSearchBase.hpp"

/// Vector of words to represent images
class BowVector: public std::map<uint32_t, double>
{
public:

	/** 
	 * Constructor
	 */
	BowVector(void);

	/**
	 * Destructor
	 */
	~BowVector(void);
	
	/**
	 * Adds a value to a word value existing in the vector, or creates a new
	 * word with the given value
	 * @param id word id to look for
	 * @param v value to create the word with, or to add to existing word
	 */
	void addWeight(uint32_t id, double v);
	
	/**
	 * Adds a word with a value to the vector only if this does not exist yet
	 * @param id word id to look for
	 * @param v value to give to the word if this does not exist
	 */
	void addIfNotExist(uint32_t id, double v);

	/**
	 * L1-Normalizes the values in the vector 
	 * @param norm_type norm used
	 */
	void normalize(LNorm norm_type);
	
	/**
	 * Prints the content of the bow vector
	 * @param out stream
	 * @param v
	 */
	friend std::ostream& operator<<(std::ostream &out, const BowVector &v);
	
	/**
	 * Saves the bow vector as a vector in a matlab file
	 * @param filename
	 * @param W number of words in the vocabulary
	 */
	void saveM(const std::string &filename, size_t W) const;
};

/// Base class of scoring functions
class GeneralScoring
{
public:
	/**
	* Computes the score between two vectors. Vectors must be sorted and 
	* normalized if necessary
	* @param v (in/out)
	* @param w (in/out)
	* @return score
	*/
	virtual double score(const BowVector &v, const BowVector &w) const = 0;

	/**
	* Returns whether a vector must be normalized before scoring according
	* to the scoring scheme
	* @param norm norm to use
	* @return true iff must normalize
	*/
	virtual bool mustNormalize(LNorm &norm) const = 0;

	/// Log of epsilon
	static const double LOG_EPS; 
	// If you change the type of WordValue, make sure you change also the
	// epsilon value (this is needed by the KL method)
	
	virtual ~GeneralScoring()
	{

	}
};
	
class  L1Scoring: public GeneralScoring 
{ 
public: 
	virtual double score(const BowVector &v, const BowVector &w) const; 
	virtual inline bool mustNormalize(LNorm &norm) const  
	{
		norm = L1; 
		return true; 
	} 
	~L1Scoring()
	{
		//no memory allocated.
	}
};
  

/// Implements a Bag of Words based (BoW) image search.  Note that search here is not implemented 
/// and would throw an error should you try to call it.  A naive implementation would have to compute
/// tf-idf with all possible image.  Instead, you should train a BoW model and
/// use this model in conjuction with a InvertedIndex search model to perform a query.
class ADTBagofWords : public ADTSearchBase 
{
public:

	/// Subclass of train params base which specifies inverted index training parameters.
	struct TrainParams : public TrainParamsBase 
	{
		TrainParams(uint32_t numClusters = 512, uint32_t numFeatures = 0) :	numClusters(numClusters), numFeatures(numFeatures) 
		{

		}
		uint32_t numClusters; // k number of clusters
		uint32_t numFeatures; // number of features to cluster
		std::string descType;
		int numberofTrainingImgs;
	};

	/// Subclass of train params base which specifies inverted index training parameters.
	struct SearchParams : public SearchParamsBase 
	{
		
	};

	/// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase 
	{
		std::vector<float> tfidf_scores;
	};

	ADTBagofWords(std::shared_ptr<TrainParams> bwPin);
	ADTBagofWords(const std::string &file_path);
	
	void compute_bow_features(const std::shared_ptr<ADTDataset> &dataset);
	/// Given a set of training parameters, list of images, trains.  Returns true if successful, false
	/// if not successful.
	bool train(const std::shared_ptr<ADTDataset> &dataset);

	/// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);

	/// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;

	/// Given a set of search parameters, a query image, searches for matching images and returns the match.
	/// Search is not valid for bag of words - this would require computing tf-idf on all possible images in the dataset, 
	/// and this function will assert(0) should you try to run it.  Instead, you should train a Bag of Words (BoW) model
	/// and use it with one of the other search mechanisms, such as the inverted index.
	std::shared_ptr<MatchResultsBase> search(const std::shared_ptr<const SearchParamsBase> &params,const std::shared_ptr<ADTDataset> &datasetTest, const std::shared_ptr<const ADTImage > &example);

	/// Returns the vocabulary matrix.
	cv::Mat vocabulary() const;

	/// Returns the number of clusters in the vocabulary.
	uint32_t num_clusters() const;
	std::string getdescType()
	{
		return descType;
	}
protected:
	std::shared_ptr<TrainParams> bwParams;
	std::string descType;
	cv::Mat vocabulary_matrix;
};
