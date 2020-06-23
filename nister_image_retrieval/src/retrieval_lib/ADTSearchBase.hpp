#pragma once

#include "ADTimages.hpp"
#include "ADTDataset.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

/// Structure to hold input training parameters (ex. #clusters, #examples to consider)
struct TrainParamsBase 
{

};

/// Structure to hold input search parameters (ex. #matches, threshold, number neighbors to consider, etc.)
struct SearchParamsBase 
{

};


/// Structure to hold returned match results (returned from a search call).
struct MatchResultsBase 
{
	std::vector<uint64_t> matches_ids;	
};


/// Abstract class from which all search structures and methods derive from.  Each search method
/// must implement train, search, load and save. 
class ADTSearchBase 
{

public:
	ADTSearchBase();
	ADTSearchBase(const std::string &file_path);

	virtual ~ADTSearchBase();

	/// Given a set of training parameters, list of images, trains.  Returns true if successful, false
	/// if not successful.
	virtual bool train (const std::shared_ptr<ADTDataset> &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const ADTImage > > &examples);

	/// Loads a trained search structure from the input filepath
	virtual bool load (const std::string &file_path) = 0;

	/// Saves a trained search structure to the input filepath
	virtual bool save (const std::string &file_path) const = 0;

	/// Given a set of search parameters, a query images, searches for matching images and returns the match
	virtual std::shared_ptr<MatchResultsBase> search(const std::shared_ptr<const SearchParamsBase> &params,const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example) = 0;

private:
	
};