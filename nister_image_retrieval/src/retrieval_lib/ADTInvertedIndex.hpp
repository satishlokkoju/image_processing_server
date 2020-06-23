#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Simple class to hold Inverted index related class for*/
/*etc.*/

#include "ADTSearchBase.hpp"
#include "ADTBOWmodel.hpp"

class ADTInvertedIndex : public ADTSearchBase 
{
public:

	struct TrainParams : public TrainParamsBase 
	{
		std::shared_ptr<ADTBagofWords> bag_of_words;  /// bag of words to index on
	};

	/// Subclass of train params base which specifies inverted index training parameters.
	struct SearchParams : public SearchParamsBase 
	{
		SearchParams(uint64_t cutoff_idx = 4096) : cutoff_idx(cutoff_idx)
		{
		
		}

		uint64_t cutoff_idx; /// number of top matches to consider
	};

	/// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase 
	{
		std::vector<float> tfidf_scores;
		double ukbenchScore;
	};

	ADTInvertedIndex();
	ADTInvertedIndex(const std::string &file_name);

	bool addtoDatabase(const std::shared_ptr<ADTDataset> &dataset, const std::shared_ptr<const TrainParamsBase> &params,const std::vector< std::shared_ptr<const ADTImage > > &examples);

	/// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);

	/// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;

	/// Returns the number of clusters used in the inverted index descriptors
	uint32_t num_clusters() const;

	std::shared_ptr<MatchResultsBase> search(const std::shared_ptr<const SearchParamsBase> &params,const std::shared_ptr<ADTDataset> &datasetTest, const std::shared_ptr<const ADTImage > &example);

protected:
	std::vector< std::vector<uint64_t> > inverted_index; /// Stores the inverted index, dimension one is the cluster index, dimension two holds a list of ids containing that word.
	std::vector<float> idf_weights; /// Stores the idf weights, one element per cluster
};

/// Prints out information about the match results.
std::ostream& operator<< (std::ostream &out, const ADTInvertedIndex::MatchResults &match_results);
