#include "ADTInvertedIndex.hpp"

#include "ADTDefines.hpp"
#include "ADTfilesystem.hpp"
#include "ADTMath.hpp"

#include <iostream>
#include <fstream>
#include <stdint.h>

ADTInvertedIndex::ADTInvertedIndex() : ADTSearchBase() 
{

}

ADTInvertedIndex::ADTInvertedIndex(const std::string &file_name) : ADTSearchBase(file_name) 
{
	if(!adtfilesystem::file_exists(file_name)) 
	{
		std::cerr << "Error reading index from " << file_name << std::endl;
		return;
	}
	
	if(!this->load(file_name)) 
	{
		std::cerr << "Error reading index from " << file_name << std::endl;
	}
}

bool ADTInvertedIndex::load (const std::string &file_path) 
{
	INFO("Reading inverted index from " + file_path + "...");
	std::ifstream ifs(file_path, std::ios::binary);
	uint32_t num_clusters;
	ifs.read((char *)&num_clusters, sizeof(uint32_t));
	inverted_index.resize(num_clusters);
	idf_weights.resize(num_clusters);
	ifs.read((char *)&idf_weights[0], sizeof(float) * num_clusters);
	for(uint32_t i=0; i<num_clusters; i++) 
	{
		uint64_t num_entries;
		ifs.read((char *)&num_entries, sizeof(uint64_t));
		inverted_index[i].resize(num_entries);
		
		if (num_entries != 0)
		  ifs.read((char *)&inverted_index[i][0], sizeof(uint64_t) * num_entries);
	}

	INFO("Done reading inverted index...");
	
	return (ifs.rdstate() & std::ifstream::failbit) == 0;
}


bool ADTInvertedIndex::save (const std::string &file_path) const 
{
	INFO( "Writing inverted index to " + file_path + "..." );

	std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);

	uint32_t num_clusters = inverted_index.size();
	ofs.write((const char *)&num_clusters, sizeof(uint32_t));
	ofs.write((const char *)&idf_weights[0], sizeof(float) * num_clusters);
	for(uint32_t i=0; i<num_clusters; i++) 
	{
		uint64_t num_entries = inverted_index[i].size();
		ofs.write((const char *)&num_entries, sizeof(uint64_t));
    
		if (num_entries != 0)
		  ofs.write((const char *)&inverted_index[i][0], sizeof(uint64_t) * num_entries);
	}

	INFO( "Done writing inverted index.");

	return (ofs.rdstate() & std::ofstream::failbit) == 0;
}

bool ADTInvertedIndex::addtoDatabase(const std::shared_ptr<ADTDataset> &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const ADTImage > > &examples) 
{
	const std::shared_ptr<const TrainParams> &ii_params = std::static_pointer_cast<const TrainParams>(params);	
	const std::shared_ptr<ADTBagofWords> &bag_of_words = ii_params->bag_of_words;
	std::string descType = bag_of_words->getdescType();

	if(!bag_of_words)
	{
		return false;
	}

	inverted_index.resize(bag_of_words->num_clusters());
	idf_weights.resize(bag_of_words->num_clusters(), 0.f);

	for (size_t i = 0; i < examples.size(); i++) 
	{
		const std::shared_ptr<const ADTImage> &image = examples[i];
		const std::string &bow_descriptors_location = dataset->feat_location(image->feature_path("bow_descriptors"));

		if (!adtfilesystem::file_exists(bow_descriptors_location)) 
			continue;

		adtmath::sparse_vector_t bow_descriptors;
		if(!adtfilesystem::load_sparse_vector(bow_descriptors_location, bow_descriptors))
			continue;

		for(size_t j=0; j<bow_descriptors.size(); j++) 
		{
			inverted_index[bow_descriptors[j].first].push_back(image->id);
		}
	}

	for(size_t i=0; i<idf_weights.size(); i++) 
	{
		idf_weights[i] = logf((float)examples.size() /(float)inverted_index[i].size());
	}

	return true;
}

std::shared_ptr<MatchResultsBase> ADTInvertedIndex::search(const std::shared_ptr<const SearchParamsBase> &params,const std::shared_ptr<ADTDataset> &datasetTest, const std::shared_ptr<const ADTImage > &example) 
{
	const std::shared_ptr<const SearchParams> &ii_params = (!params) ?std::make_shared<const SearchParams>(): std::static_pointer_cast<const SearchParams>(params);
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();
	const adtmath::sparse_vector_t &example_bow_descriptors = datasetTest->load_bow_feature(example->id);

	std::vector<std::pair<uint64_t, uint64_t> > candidates(datasetTest->num_images(), std::pair<uint64_t, uint64_t>(0, 0));
	uint64_t num_candidates = 0; // number of matches > 0
	for(size_t i=0; i<example_bow_descriptors.size(); i++) 
	{
		uint32_t cluster = example_bow_descriptors[i].first;
		for(size_t j=0; j<inverted_index[cluster].size(); j++) 
		{
			uint64_t id = inverted_index[cluster][j];
			if(!candidates[id].first) 
			{
				candidates[id].second = id;
				++num_candidates;
			}
			candidates[id].first++;
			
		}
	}

	std::sort(candidates.begin(), candidates.end());
	std::reverse(candidates.begin(), candidates.end());
	
	num_candidates = MIN(num_candidates, ii_params->cutoff_idx);

	if (num_candidates == 0)
		return match_result;

	std::vector< std::pair<float, uint64_t> > candidate_scores(num_candidates);
	for(int64_t i=0; i<num_candidates; i++) 
	{

		const adtmath::sparse_vector_t &bow_descriptors = datasetTest->load_bow_feature(candidates[i].second);

		float sim = adtmath::min_hist(example_bow_descriptors, bow_descriptors, idf_weights);
		candidate_scores[i] = std::pair<float, uint64_t>(sim, candidates[i].second);
	}

	std::sort(candidate_scores.begin(), candidate_scores.end(),boost::bind(&std::pair<float, uint64_t>::first, _1) > boost::bind(&std::pair<float, uint64_t>::first, _2));

	match_result->tfidf_scores.resize(candidate_scores.size());
	match_result->matches_ids.resize(candidate_scores.size());
	
	for(int64_t i=0; i<(int64_t)candidate_scores.size(); i++) 
	{
		match_result->tfidf_scores[i] = candidate_scores[i].first;
		match_result->matches_ids[i] = candidate_scores[i].second;
	}

	return std::static_pointer_cast<MatchResultsBase>(match_result);
}

uint32_t ADTInvertedIndex::num_clusters() const 
{
	return idf_weights.size();
}

std::ostream& operator<< (std::ostream &out, const ADTInvertedIndex::MatchResults &match_results) 
{
	out << "[ ";
	for(uint32_t i=0; i<MIN(8, match_results.matches_ids.size()); i++) 
	{
		out << "[ " << match_results.matches_ids[i] << ", " <<  match_results.tfidf_scores[i] << " ] ";
	}
	out << "]";
	return out;
}
