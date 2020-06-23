#include "ADTTreeIndex.hpp"

#include "ADTDefines.hpp"
#include "ADTfilesystem.hpp"
#include "ADTMath.hpp"
#include "ADTVision.hpp"

#include <iostream>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <memory>
#include <math.h>	// for pow
#include <utility>  // std::pair

void ADTTreeIndex::transform(const std::vector<std::vector<float>>& features, BowVector &bwv) const
{
	bwv.clear();
  
	if(m_words.empty())
	{
		INFO("Error Loading the Trained Model");
		return;
	}

	// normalize 
	LNorm norm;
	bool must = m_scoring_object->mustNormalize(norm);

	std::vector<std::vector<float>>::const_iterator fit;

	if(tiParams->wt == TF || tiParams->wt == TF_IDF)
	{
		for(fit = features.begin(); fit < features.end(); ++fit)
		{
			uint32_t id;
			double w; 
			// w is the idf value if TF_IDF, 1 if TF      
			transform(*fit, id, w);
      
			// not stopped
			if(w > 0) 
				bwv.addWeight(id, w);
		}
    
		if(!bwv.empty() && !must)
		{
			// unnecessary when normalizing
			const double nd = bwv.size();
			for(BowVector::iterator vit = bwv.begin(); vit != bwv.end(); vit++) 
				vit->second /= nd;
		}
    
	}
	else // IDF || BINARY
	{
		for(fit = features.begin(); fit < features.end(); ++fit)
		{
			uint32_t id;
			double w;
			// w is idf if IDF, or 1 if BINARY
      
			transform(*fit, id, w);
      
			// not stopped
			if(w > 0) 
				bwv.addIfNotExist(id, w);
      
		}
	} 
  
	if(must) 
		bwv.normalize(norm);
}

std::shared_ptr<MatchResultsBase> ADTTreeIndex::query(const BowVector &vec, int max_results) const
{
	BowVector::const_iterator vit;
	IFRow::const_iterator rit;
	std::shared_ptr<ADTTreeIndex::MatchResults> match_result = std::make_shared<MatchResults>();

	std::map<uint32_t, double> pairs;
	std::map<uint32_t, double>::iterator pit;
  
	for(vit = vec.begin(); vit != vec.end(); ++vit)
	{
		const uint32_t word_id = vit->first;
		const double& qvalue = vit->second;
        
		const IFRow& row = m_invfile[word_id];
    
		// IFRows are sorted in ascending entry_id order
    
		for(rit = row.begin(); rit != row.end(); ++rit)
		{
			const uint32_t entry_id = rit->entry_id;
			const double& dvalue = rit->word_weight;     
    
			double value = fabs(qvalue - dvalue) - fabs(qvalue) - fabs(dvalue);
        
			pit = pairs.lower_bound(entry_id);
			if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
			{
				pit->second += value;
			}
			else
			{
				pairs.insert(pit,std::map<uint32_t, double>::value_type(entry_id, value));
			}
      
		} // for each inverted row
	} // for each query word
	
	// move to vector
  
	typedef std::pair<uint32_t, double> matchPair;
	std::vector<matchPair> values(pairs.size());;
	uint32_t idx =0;
	for(pit = pairs.begin(); pit != pairs.end(); ++pit)
	{
		values[idx] = matchPair(pit->first, pit->second);
		idx++;
	}
	
	// resulting "scores" are now in [-2 best .. 0 worst]
	// sort vector in ascending order of score
	std::sort(values.begin(), values.end(), boost::bind(&std::pair<uint32_t, double>::second, _1) <boost::bind(&std::pair<uint32_t, double>::second, _2));
  
	// (ret is inverted now --the lower the better--)

	// cut vector
	if(max_results > 0 && (int)values.size() > max_results)
		values.resize(max_results);

	double score = 0;
	
	for (uint32_t i = 0; i < values.size(); i++) 
	{
		match_result->matches_ids.push_back(values[i].first);
		match_result->tfidf_scores.push_back(values[i].second);
	}
	
	match_result->ukbenchScore = score;
	
	return (std::shared_ptr<MatchResultsBase>)match_result;
}

// ---------------------------------------------------------------------------

uint32_t ADTTreeIndex::add(const BowVector &v,uint64_t id)
{
	uint32_t entry_id =id;
	m_nentries++;
	BowVector::const_iterator vit;
	std::vector<uint32_t>::const_iterator iit;
  
	// update inverted file
	for(vit = v.begin(); vit != v.end(); ++vit)
	{
		const uint32_t& word_id = vit->first;
		const double& word_weight = vit->second;
    
		IFRow& ifrow = m_invfile[word_id];
		ifrow.push_back(IFPair(entry_id, word_weight));
	}
  
	return entry_id;
}

ADTTreeIndex::ADTTreeIndex(std::shared_ptr<TITrainParams> tip) : ADTSearchBase() 
{
	tiParams = tip;
	m_scoring_object = new L1Scoring;
}

// struct used for writing and reading cv::mat's
struct cvmat_header 
{
	uint64_t elem_size;
	int32_t elem_type;
	uint32_t rows, cols;
};

bool ADTTreeIndex::load (const std::string &file_path) 
{
	INFO("Reading vocab tree from " + file_path + "...");

	std::ifstream ifs(file_path, std::ios::binary);
	ifs.read((char *)&tiParams->split, sizeof(uint32_t));
	ifs.read((char *)&tiParams->depth, sizeof(uint32_t));
	ifs.read((char *)&numberOfNodes, sizeof(uint32_t));

	weights.resize(numberOfNodes);
	ifs.read((char *)&weights[0], sizeof(float)*numberOfNodes);

	// load inveted files
	uint32_t invertedFileCount;
	ifs.read((char *)&invertedFileCount, sizeof(uint32_t));
	invertedFiles.resize(invertedFileCount);

	for (uint32_t i = 0; i < invertedFileCount; i++) 
	{
		uint32_t size;
		ifs.read((char *)&size, sizeof(uint32_t));
		for (uint32_t j = 0; j < size; j++) 
		{
			uint64_t imageId;
			uint32_t imageCount;
			ifs.read((char *)&imageId, sizeof(uint64_t));
			ifs.read((char *)&imageCount, sizeof(uint32_t));
			invertedFiles[i][imageId] = imageCount;
		}
	}

	// read in tree
	tree.resize(numberOfNodes);
	for (uint32_t i = 0; i < numberOfNodes; i++) 
	{
		ifs.read((char *)&tree[i].firstChildIndex, sizeof(uint32_t));
		ifs.read((char *)&tree[i].index, sizeof(uint32_t));
		ifs.read((char *)&tree[i].invertedFileLength, sizeof(uint32_t));
		ifs.read((char *)&tree[i].level, sizeof(uint32_t));
		ifs.read((char *)&tree[i].levelIndex, sizeof(uint32_t));

		// read cv::mat, copied from filesystem.cxx
		cvmat_header h;
		ifs.read((char *)&h, sizeof(cvmat_header));
		tree[i].mean.create(h.rows, h.cols, h.elem_type);

		if (h.rows == 0 || h.cols == 0) 
			continue;
		ifs.read((char *)tree[i].mean.ptr(), h.rows * h.cols * h.elem_size);
	}

	INFO("Done reading vocab tree.");
	return (ifs.rdstate() & std::ifstream::failbit) == 0;
}

bool ADTTreeIndex::save (const std::string &file_path) const 
{
	INFO("Writing vocab tree to " + file_path + "...");

	std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);

	//uint32_t num_clusters = inverted_index.size();
	ofs.write((const char *)&tiParams->split, sizeof(uint32_t));
	ofs.write((const char *)&tiParams->depth, sizeof(uint32_t));
	ofs.write((const char *)&numberOfNodes, sizeof(uint32_t));
	ofs.write((const char *)&weights[0], sizeof(float)*numberOfNodes); // weights

	// write out inverted files
	uint32_t numInvertedFiles = invertedFiles.size();
	ofs.write((const char *)&numInvertedFiles, sizeof(uint32_t));
	//for (std::unordered_map<uint64_t, uint32_t> invFile : invertedFiles) {
	//for (std::unordered_map<uint64_t, uint32_t>::iterator it = invertedFiles.begin; it != invertedFiles.end(); it++) {
	for (uint32_t i = 0; i < invertedFiles.size(); i++) 
	{
		std::unordered_map<uint64_t, uint32_t> invFile = invertedFiles[i];
		uint32_t size = invFile.size();
		ofs.write((const char *)&size, sizeof(uint32_t));
		//for (std::pair<uint64_t, uint32_t> pair : invFile) {
		for (std::unordered_map<uint64_t, uint32_t>::iterator it = invFile.begin(); it != invFile.end(); it++) 
		{
			ofs.write((const char *)&it->first, sizeof(uint64_t));
			ofs.write((const char *)&it->second, sizeof(uint32_t));
		}
	}

	// write out tree
	for (uint32_t i = 0; i < numberOfNodes; i++) 
	{
		TreeNode t = tree[i];
		ofs.write((const char *)&t.firstChildIndex, sizeof(uint32_t));
		ofs.write((const char *)&t.index, sizeof(uint32_t));
		ofs.write((const char *)&t.invertedFileLength, sizeof(uint32_t));
		ofs.write((const char *)&t.level, sizeof(uint32_t));
		ofs.write((const char *)&t.levelIndex, sizeof(uint32_t));

		// write cv::mat, copied from filesystem.cxx
		cvmat_header h;
		h.elem_size = t.mean.elemSize();
		h.elem_type = t.mean.type();
		h.rows = t.mean.rows;
		h.cols = t.mean.cols;
		ofs.write((char *)&h, sizeof(cvmat_header));
		ofs.write((char *)t.mean.ptr(), h.rows * h.cols * h.elem_size);
	}

	INFO("Done writing vocab tree.");

	return (ofs.rdstate() & std::ofstream::failbit) == 0;
}
// ----------------------------------------------------------------------------

void ADTTreeIndex::changeStructure(const std::vector<float> &plain, std::vector<std::vector<float> > &out, int L)
{
	out.resize(plain.size() / L);

	uint32_t j = 0;
	for(uint32_t i = 0; i < plain.size(); i += L, ++j)
	{
		out[j].resize(L);
		std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
	}
}


void ADTTreeIndex::loadFeatures(const std::shared_ptr<ADTDataset> &dataset,const std::vector< std::shared_ptr<const ADTImage > > &trainingImages,std::vector<std::vector<std::vector<float> > > &features)
{
    features.clear();
	
	try
	{
		features.reserve(trainingImages.size());
  	}
	catch(std::bad_alloc ex)
	{
		INFO("Unable to allocated memory.");
	}

	for(uint64_t i = 0; i <trainingImages.size(); i++)
	{
		const std::string & feat_descriptor_location = dataset->feat_location(trainingImages[i]->feature_path("descriptors"));

		cv::Mat descriptorsf,descriptors;

		if(!adtfilesystem::file_exists(feat_descriptor_location))
			continue;

		if(!adtfilesystem::load_cvmat(feat_descriptor_location, descriptors))
			continue;
		
		descriptors.convertTo(descriptorsf, CV_32FC1);
		std::vector<float> plainDesc= adtmath::dense1d(descriptorsf);

		features.push_back(std::vector<std::vector<float> >());
		changeStructure(plainDesc, features.back(), descriptorsf.cols);
	}
}

void ADTTreeIndex::getFeatures( const std::vector<std::vector<std::vector<float>> > &training_features, std::vector<const std::vector<float> *> &features) const
{
	features.resize(0);
	std::vector<std::vector<std::vector<float>> >::const_iterator vvit;
	std::vector<std::vector<float>>::const_iterator vit;

	for(vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
	{
		try
		{
			features.reserve(features.size() + vvit->size());
		}
		catch(std::bad_alloc ex)
		{
			INFO("Unable to allocated memory.");
		}

		for(vit = vvit->begin(); vit != vvit->end(); ++vit)
		{
			features.push_back(&(*vit));
		}
	}
}
  
inline double distanceLocal(const std::vector<float> &a, const std::vector<float> &b)
{
	double sqd = 0.;
	int size = a.size();
	int sizenumberoffours = size/4;
	int remainder = size%4;

	for(int i = 0; i <sizenumberoffours*4; i += 4)
	{
		sqd += (a[i  ] - b[i  ])*(a[i  ] - b[i  ]);
		sqd += (a[i+1] - b[i+1])*(a[i+1] - b[i+1]);
		sqd += (a[i+2] - b[i+2])*(a[i+2] - b[i+2]);
		sqd += (a[i+3] - b[i+3])*(a[i+3] - b[i+3]);
	}
  
	for(int i = sizenumberoffours*4; i < size; i ++)
	{
		sqd += (a[i] - b[i])*(a[i] - b[i]);
	}

	return sqd;
}

void ADTTreeIndex::transform(const std::vector<float> &feature,uint32_t &word_id, double &weight, uint32_t *nid, int levelsup) const
{ 
	// propagate the feature down the tree
	std::vector<uint32_t> nodes;
	std::vector<uint32_t>::const_iterator nit;

	// level at which the node must be stored in nid, if given
	const int nid_level = tiParams->depth - levelsup;
	if(nid_level <= 0 && nid != NULL) 
		*nid = 0; // root

	uint32_t final_id = 0; // root
	int current_level = 0;

	do
	{
		++current_level;
		nodes = m_nodes[final_id].children;
		final_id = nodes[0];
 
		double best_d = distanceLocal(feature, m_nodes[final_id].descriptor);

		for(nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
		{
		  uint32_t id = *nit;
		  double d = distanceLocal(feature, m_nodes[id].descriptor);
		  if(d < best_d)
		  {
			best_d = d;
			final_id = id;
		  }
		}
    
		if(nid != NULL && current_level == nid_level)
			*nid = final_id;
    
	} while( !m_nodes[final_id].isLeaf() );

	// turn node id into word id
	word_id = m_nodes[final_id].word_id;
	weight = m_nodes[final_id].weight;
}

void ADTTreeIndex::setNodeWeights(const std::vector<std::vector<std::vector<float>> > &training_features)
{
	const uint32_t total_visual_words = m_words.size();
	const uint32_t total_number_images = training_features.size();

	if(tiParams->wt == TF || tiParams->wt == BINARY)
	{
		// idf part must be 1 always
		for(uint32_t i = 0; i < total_visual_words; i++)
			m_words[i]->weight = 1;
	}
	else if(tiParams->wt == IDF || tiParams->wt == TF_IDF)
	{
	// IDF and TF-IDF: we calculte the idf path now

	// Note: this actually calculates the idf part of the tf-idf score.
	// The complete tf-idf score is calculated in ::transform

	std::vector<uint32_t> Ni(total_visual_words, 0);
	std::vector<bool> counted(total_visual_words, false);
    
	std::vector<std::vector<std::vector<float>> >::const_iterator mit;
	std::vector<std::vector<float>>::const_iterator fit;

	for(mit = training_features.begin(); mit != training_features.end(); ++mit)
	{
		fill(counted.begin(), counted.end(), false);

		for(fit = mit->begin(); fit < mit->end(); ++fit)
		{
		uint32_t word_id;
		double weight;
		transform(*fit, word_id,weight);

		if(!counted[word_id])
		{
			Ni[word_id]++;
			counted[word_id] = true;
		}
		}
	}

	// setting the inverse document frequency.
	for(uint32_t i = 0; i < total_visual_words; i++)
	{
		if(Ni[i] > 0)
		{
			m_words[i]->weight = log((double)total_number_images / (double)Ni[i]);
		}
	}
  }
}

inline void meanValue(const std::vector<const std::vector<float> *> &descriptors,std::vector<float> &mean,int descripSize)
{
	mean.resize(0);
	mean.resize(descripSize, 0);
  
	float s = descriptors.size();
  
	std::vector<const std::vector<float> *>::const_iterator it;
	for(it = descriptors.begin(); it != descriptors.end(); ++it)
	{
		const std::vector<float> &desc = **it;
		int sizemultiplesoffour = descripSize/4;
		for(int i = 0; i < sizemultiplesoffour*4; i += 4)
		{
			mean[i  ] += desc[i  ] / s;
			mean[i+1] += desc[i+1] / s;
			mean[i+2] += desc[i+2] / s;
			mean[i+3] += desc[i+3] / s;
		}
	
		for(int i = sizemultiplesoffour*4; i < descripSize ; i++)
		{
			mean[i] += desc[i] / s;
		}
	}
}

bool ADTTreeIndex::save2(const std::string &filename) const
{
	adtfilesystem::create_file_directory(filename);
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
	if(!fs.isOpened()) 
		throw std::string("Could not open file ") + filename;
	save(fs);

	return true;
}


void ADTTreeIndex::addImagetoUserDatabase(const std::shared_ptr<ADTDataset> &dataset,std::shared_ptr<const ADTImage > image)
{	
	const std::string & keypoints_location =  dataset->feat_location(image->feature_path("keypoints"));
	const std::string & descriptors_location =  dataset->feat_location(image->feature_path("descriptors"));

	cv::Mat im = cv::imread(image->get_location(), cv::IMREAD_GRAYSCALE);
	cv::Mat keypoints, descriptors,descriptorsf;
		
	if(!ADTVision::compute_sparse_akaze_feature(im,keypoints,descriptors)) 
		return;

	adtfilesystem::create_file_directory(keypoints_location);
	adtfilesystem::create_file_directory(descriptors_location);

	adtfilesystem::write_cvmat(keypoints_location, keypoints);
	adtfilesystem::write_cvmat(descriptors_location, descriptors);

	if(!adtfilesystem::file_exists(descriptors_location))
	{
		INFO("features not calculated");
		return;
	}

	if(!adtfilesystem::load_cvmat(descriptors_location, descriptors))
	{
		INFO("Unable to loat cv::Mat to Memory");
		return;
	}

	descriptors.convertTo(descriptorsf, CV_32FC1);
	std::vector<float> plainDesc= adtmath::dense1d(descriptorsf);
	std::vector<std::vector<float> > features;
	changeStructure(plainDesc, features, descriptors.cols);

	addImagetoDatabase(features,image->id);
}

void ADTTreeIndex::addImagetoDatabase(std::vector<std::vector<float> > &featureV,uint64_t id)
{
	BowVector v;
	transform(featureV,v);
	add(v,id);
}

bool ADTTreeIndex::addtoDatabase(const std::shared_ptr<ADTDataset> &dataset,const std::vector< std::shared_ptr<const ADTImage > > &imagesToBeadded)
{    
	// resize vectors
	m_invfile.resize(0);
	m_invfile.resize(m_words.size());
	m_nentries = 0;

	if(imagesToBeadded.size() <=0)
	{
		INFO("The search database is empty");
		exit(1);
	}

	for(uint64_t i = 0; i < imagesToBeadded.size(); i++)
	{
		const std::string & surf_descriptor_location = dataset->feat_location(imagesToBeadded[i]->feature_path("descriptors"));

		cv::Mat descriptors,descriptorsf;

		if(!adtfilesystem::file_exists(surf_descriptor_location))
			continue;

		if(!adtfilesystem::load_cvmat(surf_descriptor_location, descriptors))
			continue;
		
		descriptors.convertTo(descriptorsf, CV_32FC1);
		std::vector<float> plainDesc= adtmath::dense1d(descriptorsf);
		std::vector<std::vector<float> > features;
		changeStructure(plainDesc, features, descriptors.cols);

		addImagetoDatabase(features,imagesToBeadded[i]->id);
	}

	return true;
}
// --------------------------------------------------------------------------

inline std::string toStringLocal(const std::vector<float> &a)
{
	std::stringstream ss;
  
	for(unsigned int i = 0; i < a.size(); ++i)
	{
		ss << a[i] << " ";
	}

	return ss.str();
}

// --------------------------------------------------------------------------
  
inline void fromStringLocal(std::vector<float> &a, const std::string &s,uint32_t descSize)
{
  a.resize(descSize);
  
  std::stringstream ss(s);
  for(uint32_t i = 0; i < descSize; ++i)
  {
    ss >> a[i];
  }
}

void ADTTreeIndex::save(cv::FileStorage &f,const std::string &name) const
{
  // Format YAML:
  // vocabulary 
  // {
  //   k:
  //   L:
  //   scoringType:
  //   weightingType:
  //   nodes 
  //   [
  //     {
  //       nodeId:
  //       parentId:
  //       weight:
  //       descriptor: 
  //     }
  //   ]
  //   words
  //   [
  //     {
  //       wordId:
  //       nodeId:
  //     }
  //   ]
  // }
  //
  // The root node (index 0) is not included in the node vector
  //
  
  f << name << "{";
  
  f << "k" << (int)tiParams->split;
  f << "L" << (int)tiParams->depth;
  f << "scoringType" << (int)tiParams->st;
  f << "weightingType" << (int)tiParams->wt;
  f << "descSize" <<(int)descriptorSize;
  // tree
  f << "nodes" << "[";
  std::vector<uint32_t> parents, children;
  std::vector<uint32_t>::const_iterator pit;

  parents.push_back(0); // root

  while(!parents.empty())
  {
    uint32_t pid = parents.back();
    parents.pop_back();

    const TreeNode& parent = m_nodes[pid];
    children = parent.children;

    for(pit = children.begin(); pit != children.end(); pit++)
    {
      const TreeNode& child = m_nodes[*pit];

      // save node data
      f << "{:";
      f << "nodeId" << (int)child.id;
      f << "parentId" << (int)pid;
      f << "weight" << (double)child.weight;
      f << "descriptor" << toStringLocal(child.descriptor);
      f << "}";
      
      // add to parent list
      if(!child.isLeaf())
      {
        parents.push_back(*pit);
      }
    }
  }
  
  f << "]"; // nodes

  // words
  f << "words" << "[";
  
  std::vector<TreeNode*>::const_iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); wit++)
  {
    uint32_t id = wit - m_words.begin();
    f << "{:";
    f << "wordId" << (int)id;
    f << "nodeId" << (int)(*wit)->id;
    f << "}";
  }
  
  f << "]"; // words

  f << "}";

}

void ADTTreeIndex::load2(const std::string &file_name)
{
	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
	
	if(!fs.isOpened())
		throw std::string("Could not open file ") + file_name;

	std::string name = "vocabulary";
	m_words.clear();
	m_nodes.clear();
  
	cv::FileNode fvoc = fs[name];
  
	tiParams->split = (int)fvoc["k"];
	tiParams->depth = (int)fvoc["L"];
	tiParams->st = (ScoringType)((int)fvoc["scoringType"]);
	tiParams->wt = (WeightingType)((int)fvoc["weightingType"]);
	uint32_t descSize=  (int)fvoc["descSize"];
  	// nodes
	cv::FileNode fn = fvoc["nodes"];

	m_nodes.resize(fn.size() + 1); // +1 to include root
	m_nodes[0].id = 0;

	for(unsigned int i = 0; i < fn.size(); ++i)
	{
		uint32_t nid = (int)fn[i]["nodeId"];
		uint32_t pid = (int)fn[i]["parentId"];
		double weight = (double)fn[i]["weight"];
		std::string d = (std::string)fn[i]["descriptor"];
    
		m_nodes[nid].id = nid;
		m_nodes[nid].parent = pid;
		m_nodes[nid].weight = weight;
		m_nodes[pid].children.push_back(nid);
    
		fromStringLocal(m_nodes[nid].descriptor, d,descSize);
	}
  
	// words
	fn = fvoc["words"];  
	m_words.resize(fn.size());

	for(unsigned int i = 0; i < fn.size(); ++i)
	{
		uint32_t wid = (int)fn[i]["wordId"];
		uint32_t nid = (int)fn[i]["nodeId"];
    
		m_nodes[nid].word_id = wid;
		m_words[wid] = &m_nodes[nid];
	}
}

bool ADTTreeIndex::save_db_woid(const std::string &filename,int image_id) const
{
	adtfilesystem::create_file_directory(filename);
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if(!fs.isOpened()) 
		throw std::string("Could not open file ") + filename;
    // Format YAML:
	// vocabulary { ... see TemplatedVocabulary::save }
	// database 
	// {
	//   nEntries: 
	//   usingDI: 
	//   diLevels: 
	//   invertedIndex
	//   [
	//     [
	//        { 
	//          imageId: 
	//          weight: 
	//        }
	//     ]
	//   ]
	//   directIndex
	//   [
	//      [
	//        {
	//          nodeId:
	//          features: [ ]
	//        }
	//      ]
	//   ]

	// invertedIndex[i] is for the i-th word
	// directIndex[i] is for the i-th entry
	// directIndex may be empty if not using direct index
	//
	// imageId's and nodeId's must be stored in ascending order
	// (according to the construction of the indexes)

	fs <<  "index" << "{";
  
	fs << "nEntries" << m_nentries;
  
	fs << "invertedIndex" << "[";
  
	InvertedFile::const_iterator iit;
	IFRow::const_iterator irit;

	for(iit = m_invfile.begin(); iit != m_invfile.end(); ++iit)
	{
		fs << "["; // word of IF
		for(irit = iit->begin(); irit != iit->end(); ++irit)
		{
			if(irit->entry_id !=image_id)
			{
				fs << "{:" 
					<< "imageId" << (int)irit->entry_id
					<< "weight" << irit->word_weight
					<< "}"; 
			}

		}
		fs << "]"; // word of IF
	}
  
	fs << "]"; // invertedIndex
	fs << "}"; // database

	return true;
}

bool ADTTreeIndex::save_db(const std::string &filename) const
{
	adtfilesystem::create_file_directory(filename);
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if(!fs.isOpened()) 
		throw std::string("Could not open file ") + filename;
  
	return save_db(fs);
}

bool ADTTreeIndex::init_db_memory(const std::string &filename)
{
  	// resize vectors
	m_invfile.resize(0);
	m_invfile.resize(m_words.size());
	m_nentries = 0;

	return true;
}

bool ADTTreeIndex::load_db(const std::string &filename)
{
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
	if(!fs.isOpened())
		throw std::string("Could not open file ") + filename;
  
	return this->load_db(fs);
}

bool ADTTreeIndex::load_db(const cv::FileStorage &fs,const std::string &name)
{
	// load database now
	m_invfile.clear();
	cv::FileNode fdb = fs[name];
  
	m_nentries = (int)fdb["nEntries"]; 
	cv::FileNode fn = fdb["invertedIndex"];
	m_invfile.resize(fn.size());
	for(uint32_t wid = 0; wid < fn.size(); ++wid)
	{
		cv::FileNode fw = fn[wid];
    
		for(unsigned int i = 0; i < fw.size(); ++i)
		{
			uint32_t eid = (int)fw[i]["imageId"];
			double v = fw[i]["weight"];
      
			m_invfile[wid].push_back(IFPair(eid, v));
		}
	}
	
	return true;
}


bool ADTTreeIndex::save_db(cv::FileStorage &fs,const std::string &name) const
{
  // Format YAML:
  // vocabulary { ... see TemplatedVocabulary::save }
  // database 
  // {
  //   nEntries: 
  //   usingDI: 
  //   diLevels: 
  //   invertedIndex
  //   [
  //     [
  //        { 
  //          imageId: 
  //          weight: 
  //        }
  //     ]
  //   ]
  //   directIndex
  //   [
  //      [
  //        {
  //          nodeId:
  //          features: [ ]
  //        }
  //      ]
  //   ]

  // invertedIndex[i] is for the i-th word
  // directIndex[i] is for the i-th entry
  // directIndex may be empty if not using direct index
  //
  // imageId's and nodeId's must be stored in ascending order
  // (according to the construction of the indexes)

	fs << name << "{";
  
	fs << "nEntries" << m_nentries;
  
	fs << "invertedIndex" << "[";
  
	InvertedFile::const_iterator iit;
	IFRow::const_iterator irit;

	for(iit = m_invfile.begin(); iit != m_invfile.end(); ++iit)
	{
		fs << "["; // word of IF
		for(irit = iit->begin(); irit != iit->end(); ++irit)
		{
			fs << "{:" 
			<< "imageId" << (int)irit->entry_id
			<< "weight" << irit->word_weight
			<< "}";
		}
		fs << "]"; // word of IF
	}
  
	fs << "]"; // invertedIndex
	fs << "}"; // database

	return true;
}


bool ADTTreeIndex::train2(const std::shared_ptr<ADTDataset> &dataset) 
{
	m_nodes.clear();
	m_words.clear();
	int numbofTraining = tiParams->numberofTrainingImgs;
	
	if(numbofTraining > dataset->all_images().size())
	{
		INFO("Number of training images is more than the Training database size");
		INFO("Add more images to the training folder and rerun");
		exit(1);
	}

	std::vector<std::shared_ptr<const ADTImage> > random_Training_images = dataset->random_images(numbofTraining);
	// expected_nodes = Sum_{i=0..L} ( k^i )
	int expected_nodes = (int)((pow((double)tiParams->split, (double)tiParams->depth + 1) - 1)/(tiParams->split - 1));
	try
	{
		m_nodes.reserve(expected_nodes);
	}
	catch(std::bad_alloc ex)
	{
		INFO("Unable to allocated memory.");
	}
  
	// avoid allocations when creating the tree
	std::vector<std::vector<std::vector<float> > > features;
	INFO("Loading Features started");
	loadFeatures(dataset,random_Training_images,features);
	INFO("Loading Features complete");
	std::vector<const std::vector<float> *> featuresp;

	INFO("Convert Features started");
	getFeatures(features, featuresp);
	INFO("Convert Features Complete");

    descriptorSize = featuresp[0]->size();
	// create root  
	m_nodes.push_back(TreeNode(0)); // root
  
	// create the tree
	INFO("Creating Vocabulary Tree...");
	HKmeansStep(0, featuresp, 1);
	
	INFO("Creating Vocabulary Complete.");
	// create the words
	createWords();

	// and set the weight of each node of the tree
	setNodeWeights(features);

	return true;
}

void ADTTreeIndex::HKmeansStep(uint32_t parent_id, const std::vector<const std::vector<float> *> &descriptors, int current_level)
{
  if(descriptors.empty()) 
	  return;

  std::stringstream ss;
  ss.str("");
  ss << "Kmeans clustering...";
  ss << current_level;

  INFO(ss.str());
  // features associated to each cluster
  std::vector<std::vector<float> > clusters;
  std::vector<std::vector<uint32_t> > groups; 
  
  // groups[i] = [j1, j2, ...]
  // j1, j2, ... indices of descriptors associated to cluster i

    try
    {
		clusters.reserve(tiParams->split);
		groups.reserve(tiParams->split);
	}
	catch(std::bad_alloc ex)
	{
		INFO("Unable to allocated memory.");
	}
  //const int msizes[] = { tiParams->split, descriptors.size() };
  //cv::SparseMat assoc(2, msizes, CV_8U);
  //cv::SparseMat last_assoc(2, msizes, CV_8U);  
  //assoc.row(cluster_idx).col(descriptor_idx) = 1 iif associated
  
  if((int)descriptors.size() <= tiParams->split)
  {
    // trivial case: one cluster per feature
    groups.resize(descriptors.size());

    for(unsigned int i = 0; i < descriptors.size(); i++)
    {
      groups[i].push_back(i);
      clusters.push_back(*descriptors[i]);
    }
  }
  else
  {
    // select clusters and groups with kmeans
    
    bool first_time = true;
    bool goon = true;
    
    // to check if clusters move after iterations
    std::vector<int> last_association, current_association;

    while(goon)
    {
			// 1. Calculate clusters
		if(first_time)
		{
			// random sample 
			initiateClusters(descriptors, clusters);
		}
		else
		{
			// calculate cluster centres
			for(unsigned int c = 0; c < clusters.size(); ++c)
			{
				std::vector<const std::vector<float> *> cluster_descriptors;
				try
				{
					cluster_descriptors.reserve(groups[c].size());
				}
				catch(std::bad_alloc ex)
				{
					INFO("Unable to allocated memory.");
				}
							/*
				for(unsigned int d = 0; d < descriptors.size(); ++d)
				{
				if( assoc.find<unsigned char>(c, d) )
				{
					cluster_descriptors.push_back(descriptors[d]);
				}
				}
				*/
          
				std::vector<unsigned int>::const_iterator vit;
				for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)
				{
					cluster_descriptors.push_back(descriptors[*vit]);
				}

				meanValue(cluster_descriptors, clusters[c],descriptors[0]->size());
			}
      } 

	  // if(!first_time)
      // 2. Associate features with clusters
      // calculate distances to cluster centers

      groups.clear();
      groups.resize(clusters.size(), std::vector<uint32_t>());
      current_association.resize(descriptors.size());
      std::vector<const std::vector<float> *>::const_iterator fit;

      //unsigned int d = 0;
      for(fit = descriptors.begin(); fit != descriptors.end(); ++fit)//, ++d)
      {
        double best_dist = distanceLocal(*(*fit), clusters[0]);
        unsigned int icluster = 0;
        
        for(unsigned int c = 1; c < clusters.size(); ++c)
        {
          double dist = distanceLocal(*(*fit), clusters[c]);
          if(dist < best_dist)
          {
            best_dist = dist;
            icluster = c;
          }
        }

        //assoc.ref<unsigned char>(icluster, d) = 1;

        groups[icluster].push_back(fit - descriptors.begin());
        current_association[ fit - descriptors.begin() ] = icluster;
      }
      
      // kmeans++ ensures all the clusters has any feature associated with them

      // 3. check convergence
      if(first_time)
      {
        first_time = false;
      }
      else
      {
        //goon = !eqUChar(last_assoc, assoc);
        
        goon = false;
        for(unsigned int i = 0; i < current_association.size(); i++)
        {
          if(current_association[i] != last_association[i])
		  {
            goon = true;
            break;
          }
        }
      }

		if(goon)
		{
			// copy last feature-cluster association
			last_association = current_association;
			//last_assoc = assoc.clone();
		}
			
	} // while(goon)
  
  } // if must run kmeans
  
  // create nodes
  for(uint32_t i = 0; i < clusters.size(); ++i)
  {
    uint32_t id = m_nodes.size();
    m_nodes.push_back(TreeNode(id));
    m_nodes.back().descriptor = clusters[i];
    m_nodes.back().parent = parent_id;
    m_nodes[parent_id].children.push_back(id);
  }
  
  // go on with the next level
  if(current_level < tiParams->depth)
  {
    // iterate again with the resulting clusters
    const std::vector<uint32_t> &children_ids = m_nodes[parent_id].children;
    for(unsigned int i = 0; i < clusters.size(); ++i)
    {
		uint32_t id = children_ids[i];
		std::vector<const std::vector<float> *> child_features;
		try
		{
			child_features.reserve(groups[i].size());
		}
		catch(std::bad_alloc ex)
		{
		INFO("Unable to allocated memory.");
		}
		
		std::vector<uint32_t>::const_iterator vit;
		
		for(vit = groups[i].begin(); vit != groups[i].end(); ++vit)
		{
			child_features.push_back(descriptors[*vit]);
		}

		if(child_features.size() > 1)
		{
			HKmeansStep(id, child_features, current_level + 1);
		}
    }
  }
}

// --------------------------------------------------------------------------

void ADTTreeIndex::initiateClusters(const std::vector<const std::vector<float> *> &descriptors, std::vector<std::vector<float> > &clusters) const
{
	initiateClustersKMpp(descriptors, clusters);  
}

// --------------------------------------------------------------------------

void ADTTreeIndex::initiateClustersKMpp(const std::vector<const std::vector<float> *> &pfeatures, std::vector<std::vector<float> > &clusters) const
{
	// Implements kmeans++ seeding algorithm
	// Algorithm:
	// 1. Choose one center uniformly at random from among the data points.
	// 2. For each data point x, compute D(x), the distance between x and the nearest 
	//    center that has already been chosen.
	// 3. Add one new data point as a center. Each point x is chosen with probability 
	//    proportional to D(x)^2.
	// 4. Repeat Steps 2 and 3 until k centers have been chosen.
	// 5. Now that the initial centers have been chosen, proceed using standard k-means 
	//    clustering.

	boost::mt19937 rng(std::time(0)); 
	boost::uniform_int<> zerotoPfeaturessize( 0, pfeatures.size()-1 );
    boost::variate_generator< boost::mt19937, boost::uniform_int<> > randboostint(rng,zerotoPfeaturessize);

	clusters.resize(0);
	try
	{
		clusters.reserve(tiParams->split);
	}
	catch(std::bad_alloc ex)
	{
		INFO("Unable to allocated memory.");
	}
	std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());
  
	// 1.
  
	int ifeature = randboostint();
  
	// create first cluster
	clusters.push_back(*pfeatures[ifeature]);

	// compute the initial distances
	std::vector<const std::vector<float> *>::const_iterator fit;
	std::vector<double>::iterator dit;
	dit = min_dists.begin();
	for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
	{
		*dit = distanceLocal(*(*fit), clusters.back());
	}  

	while((int)clusters.size() < tiParams->split)
	{
		// 2.
		dit = min_dists.begin();
		for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
		{
			if(*dit > 0)
			{
				double dist = distanceLocal(*(*fit), clusters.back());
				if(dist < *dit) 
					*dit = dist;
			}
		}
    
		// 3.
		double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

		if(dist_sum > 0)
		{
			double cut_d;
			do
			{  
				boost::mt19937 rngDouble(std::time(0)); 
				boost::uniform_real<> zerotodist_sum( 0, dist_sum );
				boost::variate_generator< boost::mt19937, boost::uniform_real<> > randboostDouble(rngDouble,zerotodist_sum);
				cut_d = randboostDouble();
			} 
			while(cut_d == 0.0);

			double d_up_now = 0;
			for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
			{
				d_up_now += *dit;
				if(d_up_now >= cut_d) 
					break;
			}
      
			if(dit == min_dists.end()) 
				ifeature = pfeatures.size()-1;
			else
				ifeature = dit - min_dists.begin();
      
			clusters.push_back(*pfeatures[ifeature]);

		} // if dist_sum > 0
		else
			break;
      
	} // while(used_clusters < tiParams->split)
  
}

// --------------------------------------------------------------------------

void ADTTreeIndex::createWords()
{
	m_words.resize(0);
  
	if(!m_nodes.empty())
	{
		try
		{
			m_words.reserve( (int)pow((double)tiParams->split, (double)tiParams->depth) );
		}
		catch(std::bad_alloc ex)
		{
			INFO("Unable to allocated memory.");
		}
		std::vector<TreeNode>::iterator nit;
    
		nit = m_nodes.begin(); // ignore root
		for(++nit; nit != m_nodes.end(); ++nit)
		{
			if(nit->isLeaf())
			{
				nit->word_id = m_words.size();
				m_words.push_back( &(*nit) );
			}
		}
	}
}

bool ADTTreeIndex::train(const std::shared_ptr<ADTDataset> &dataset) 
{
	uint32_t split = tiParams->split;
	uint32_t depthLevel = tiParams->depth;
	
	int numbofTraining = tiParams->numberofTrainingImgs;
	std::vector<std::shared_ptr<const ADTImage> > random_Training_images = dataset->random_images(numbofTraining);

	numberOfNodes = (uint32_t)(pow(split, depthLevel) - 1) / (split - 1);
	weights.resize(numberOfNodes);
	tree.resize(numberOfNodes);
	invertedFiles.resize((uint32_t)pow(split, depthLevel - 1));

	// took the following from bag_of_words
	std::vector<uint64_t> all_ids(random_Training_images.size());
	for (uint32_t i = 0; i < random_Training_images.size(); i++) 
	{
		all_ids[i] = random_Training_images[i]->id;
	}

	std::random_shuffle(all_ids.begin(), all_ids.end());

	std::vector<cv::Mat> all_descriptors;
	uint64_t num_features = 0;

	std::vector<uint64_t> new_ids;

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
			descriptors.convertTo(descriptorsf, CV_32FC1);
			num_features += descriptors.rows;  
			new_ids.push_back(all_ids[i]);
			all_descriptors.push_back(descriptorsf);
		}
	}

	all_ids = new_ids;

	const cv::Mat merged_descriptor = ADTVision::merge_descriptors(all_descriptors, false);
	cv::Mat labels;
	uint32_t attempts = 1;
	cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 18, 0.000001);
	uint32_t startNode = 0;
	uint32_t startLevel = 0;
	uint32_t levelIndex = 0;

	tree[startNode].levelIndex = levelIndex;
	tree[startNode].index = startNode;
	buildTreeRecursive(startNode, merged_descriptor, tc, attempts, cv::KMEANS_PP_CENTERS, startLevel);
  
	// generate data on the reference images - descriptors go down tree, add images to inverted lists at leaves, and generate di vector for image
	// Also stores counts for how many images pass through each node to calculate weights
	std::vector<uint32_t> counts(numberOfNodes);
	for (size_t i = 0; i < numberOfNodes; i++)
		counts[i] = 0;

 
  // have to reload all descriptors, do once here instead of twice in the next 2 loops
  // all_descriptors.clear();
  // for (int i = 0; i < all_ids.size(); i++) {
  //   std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset->image(all_ids[i]));
  //   if (image == 0) continue;

  //   const std::string &descriptors_location = dataset->location(image->feature_path("descriptors"));
  //   if (!filesystem::file_exists(descriptors_location)) continue;

  //   cv::Mat descriptors, descriptorsf;
  //   if (filesystem::load_cvmat(descriptors_location, descriptors)) {
  //     descriptors.convertTo(descriptorsf, CV_32FC1);
  //     num_features += descriptors.rows;

  //     all_descriptors.push_back(descriptorsf);
  //   }
  // }

  for (uint32_t i = 0; i < all_ids.size(); i++) 
  {
	std::vector<float> result = generateVector(all_descriptors[i], false, true, false, all_ids[i]);

	// accumulate counts
	for (size_t j = 0; j < numberOfNodes; j++)
	{
		if (result[j] > 0)
	#pragma omp critical
		{
			counts[j]++;
		}
	}
  }
    
  // create weights according to equation 4: w_i = ln(N / N_i)
  for (int i = 0; i < numberOfNodes; i++) 
  {
    if (counts[i] == 0)
      weights[i] = 0;
    else
      weights[i] = log(((float)all_ids.size()) / ((float)counts[i]));
  }

  // generate datavectors, normalize, then write to disk
  for (uint32_t i = 0; i < all_ids.size(); i++) 
  {

    std::vector<float> dataVec = generateVector(all_descriptors[i], true, true, false, all_ids[i]);
  
	float length = 0; 
	
	// hopefully shouldn't overflow from adding doubles
    //std::vector<float> datavec = (iterator->second);
    
	for (size_t j = 0; j < numberOfNodes; j++) 
	{
      //(iterator->second)[i] *= weights[i];
      length += (float)pow(dataVec[j], 2.0);
    }
    
	// normalizing
    length = sqrt(length);
    for (size_t j = 0; j < numberOfNodes; j++) 
      dataVec[j] /= length;

	// write out vector to database
	std::shared_ptr<ADTImage> image = std::static_pointer_cast<ADTImage>(dataset->image(all_ids[i]));
	const std::string &datavec_location = dataset->feat_location(image->feature_path("datavec"));
	adtfilesystem::create_file_directory(datavec_location);
    
	if(!adtfilesystem::write_vector(datavec_location, dataVec)) 
	{
		std::cerr << "Failed to write data for " << all_ids[i] << " to " << datavec_location << std::endl;
	}
    
    // std::ofstream ofs(datavec_location.c_str(), std::ios::binary | std::ios::trunc);
    // ofs.write((char *)&dataVec[0], numberOfNodes*sizeof(float));
    // if ((ofs.rdstate() & std::ofstream::failbit) != 0)
    //   INFO("Failed to write data for " << all_ids[i] << " to " << datavec_location << std::endl;

    /*if (!filesystem::file_exists(datavec_location)) { printf("COULDN'T FIND FILE\n\n"); continue; };
    std::vector<float> dbVec(numberOfNodes);
    std::ifstream ifs(datavec_location.c_str(), std::ios::binary);
    ifs.read((char *)&dbVec[0], numberOfNodes*sizeof(float));
    if ((ifs.rdstate() & std::ifstream::failbit) != 0) { printf("FAILLLLLLLLLL\n\n"); continue; }

    printf("Original: ");
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", datavec[i]);
    printf("\nSaved: ");
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", dbVec[i]);
    printf("\n\n");*/
  }

  // for (int i = 0; i < invertedFiles.size(); i++)
  //   printf("Size %d: %d\n", i, invertedFiles[i].size());

  /*uint32_t l = 0, inL = 0;
  for (uint32_t i = 0; i < numberOfNodes; i++) {
    printf("Node %d, ifl %d, count %d, weight %f Desc (%d):\n ", i, tree[i].invertedFileLength, counts[i], weights[i], tree[i].mean.cols);
    for (int j = 0; j < tree[i].mean.cols && j<8; j++)
      printf("%f ", tree[i].mean.at<float>(0,j));
    printf("\n\n");
    inL++;
    if (inL >= (uint32_t)pow(split, l)) {
      l++;
      inL = 0;
      printf("-----------------------------------------\n\n");
    }
  }*/

    
  return true;
}


void ADTTreeIndex::buildTreeRecursive(uint32_t t, const cv::Mat &descriptors, cv::TermCriteria &tc, int attempts, int flags, int currLevel) 
{
	tree[t].invertedFileLength = descriptors.rows;
	tree[t].level = currLevel;
	// handles the leaves
	uint32_t maxLevel = tiParams->depth;
	uint32_t split = tiParams->split;

	if (currLevel == maxLevel - 1) 
	{
		tree[t].firstChildIndex = 0;
		return;
	}
  
	cv::Mat labels;
	cv::Mat centers;
  
	std::vector<cv::Mat> groups(split);
	std::vector< std::vector<cv::Mat> > unjoinedGroups(split);

	for (uint32_t i = 0; i < split; i++)
	groups[i] = cv::Mat();
 
	  bool enoughToFill = true;
	  if (descriptors.rows >= split) 
	  {
		// gather desired descriptors
		cv::kmeans(descriptors, split, labels, tc, attempts, flags, centers);

		for (uint32_t i = 0; i < labels.rows; i++) 
		{
		  int index = labels.at<int>(i);
		  unjoinedGroups[index].push_back(descriptors.row(i));
		}
	  }
	  else
	  {
		enoughToFill = false;
		for (uint32_t i = 0; i < descriptors.rows; i++) 
		{
		  unjoinedGroups[i].push_back(descriptors.row(i));
		}

	  }

	for (uint32_t i = 0; i<split; i++) 
	{
		if (unjoinedGroups[i].size() > 0) 
		{
			groups[i] = ADTVision::merge_descriptors(unjoinedGroups[i], false);
		}
	}

	for (uint32_t i = 0; i < split; i++) 
	{
		uint32_t childLevelIndex = tree[t].levelIndex*split + i;
		uint32_t childIndex = (uint32_t)((pow(split, tree[t].level + 1) - 1) / (split - 1)) + childLevelIndex;
		if (i == 0)
			tree[t].firstChildIndex = childIndex;

		if (enoughToFill)
			cv::normalize(centers.row(i), tree[childIndex].mean);
      
		tree[childIndex].levelIndex = childLevelIndex;
		tree[childIndex].index = childIndex;

		buildTreeRecursive(childIndex, groups[i], tc, attempts, flags, currLevel + 1);
	}
}

std::vector<float> ADTTreeIndex::generateVector(const cv::Mat &descriptors, bool shouldWeight, bool building, bool multinode, int64_t id) 
{
  std::unordered_set<uint32_t> dummy;
  return generateVector(descriptors, shouldWeight, building, multinode, dummy, id);
}

std::vector<float> ADTTreeIndex::generateVector(const cv::Mat &descriptors, bool shouldWeight, bool building, bool multinode, std::unordered_set<uint32_t> & possibleMatches, int64_t id) 
{
  std::vector<float> vec(numberOfNodes);
  for (uint32_t i = 0; i < numberOfNodes; i++)
    vec[i] = 0;


  for (uint32_t r = 0; r < descriptors.rows; r++) 
  {
    //printf("%d ", r);
    generateVectorHelper(0, descriptors.row(r), vec, possibleMatches, building, id);
  }

  if (shouldWeight) 
  {
    float length = 0; // for normalizing
    for (uint32_t i = 0; i < numberOfNodes; i++) 
	{
      vec[i] *= weights[i];
      length += vec[i] * vec[i];
    }

    length = sqrt(length);
    for (uint32_t i = 0; i < numberOfNodes; i++) 
	{
      if(length == 0)
        vec[i] = 0;
      else
        vec[i] /= length;
    }

  }

  return vec;
}

void ADTTreeIndex::generateVectorHelper(uint32_t nodeIndex, const cv::Mat &descriptor, std::vector<float> & counts, std::unordered_set<uint32_t> & possibleMatches, bool building, int64_t id) 
{
#pragma omp critical
    {
      counts[nodeIndex]++;
    }

	if (tree[nodeIndex].firstChildIndex <= 0) 
	{
    std::unordered_map<uint64_t, uint32_t> & invFile = invertedFiles[tree[nodeIndex].levelIndex];
    if (id >= 0) 
	{
#pragma omp critical
      {
		 if (invFile.find(id) == invFile.end())
			invFile[id] = 1;
		else
			invFile[id]++;
      }
    }
    // accumulating image id's into possibleMatches
    else if(!building)
	{
	
#pragma omp critical
      {
        possibleMatches.insert(tree[nodeIndex].levelIndex);
      }
    }
  }
  else
  {
    uint32_t maxChild = tree[nodeIndex].firstChildIndex;
    
	double max = (tree[maxChild].mean.dims == 0 || descriptor.dims == 0 || tree[maxChild].mean.type() != descriptor.type() || tree[maxChild].mean.size() != descriptor.size() ) ? 0 : descriptor.dot(tree[maxChild].mean);
  
    for (uint32_t i = 1; i < tiParams->split; i++) 
	{
      if (tree[nodeIndex].invertedFileLength == 0)
        continue;
      uint32_t childIndex = tree[nodeIndex].firstChildIndex + i;
   
      if (tree[childIndex].mean.dims == 0 || descriptor.dims == 0 || tree[childIndex].mean.type() != descriptor.type() || tree[childIndex].mean.size() !=  descriptor.size())
        continue;
      double dot = descriptor.dot(tree[childIndex].mean);

      if (dot>max) 
	  {
        max = dot;
        maxChild = childIndex;
      }
    }

    generateVectorHelper(maxChild, descriptor, counts, possibleMatches, building, id);
  }
}

std::shared_ptr<MatchResultsBase> ADTTreeIndex::search2( const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example) 
{
	const std::shared_ptr<const SearchParams> &vt_params = (!params) ?std::make_shared<const SearchParams>(): std::static_pointer_cast<const SearchParams>(params);	

	// get descriptors for example
	if (!example) 
		return std::shared_ptr<MatchResultsBase>();

	const std::string &descriptors_location = datasetTest->feat_location(example->feature_path("descriptors"));
  
	if (!adtfilesystem::file_exists(descriptors_location))
		return std::shared_ptr<MatchResultsBase>();

	cv::Mat descriptors, descriptorsf;
	if (!adtfilesystem::load_cvmat(descriptors_location, descriptors)) 
		return std::shared_ptr<MatchResultsBase>();

	std::unordered_set<uint32_t> possibleMatches;
	descriptors.convertTo(descriptorsf, CV_32FC1);
	std::vector<float> plainDesc= adtmath::dense1d(descriptorsf);
	std::vector<std::vector<float> > features;
	changeStructure(plainDesc, features, descriptors.cols);
	
	BowVector vec;
	transform(features, vec);

	return  (std::shared_ptr<MatchResultsBase>)query(vec, vt_params->amountToReturn);
}


std::shared_ptr<MatchResultsBase> ADTTreeIndex::search_user( const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example) 
{
	const std::shared_ptr<const SearchParams> &vt_params = (!params) ?std::make_shared<const SearchParams>(): std::static_pointer_cast<const SearchParams>(params);	

	// get descriptors for example
	if (!example) 
		return std::shared_ptr<MatchResultsBase>();
	
	cv::Mat im = cv::imread(example->get_location(),cv::IMREAD_GRAYSCALE);
	cv::Mat keypoints, descriptors,descriptorsf;
	
	if(!ADTVision::compute_sparse_akaze_feature(im,keypoints,descriptors)) 
		return std::shared_ptr<MatchResultsBase>();
		

	std::unordered_set<uint32_t> possibleMatches;
	descriptors.convertTo(descriptorsf, CV_32FC1);
	std::vector<float> plainDesc= adtmath::dense1d(descriptorsf);
	std::vector<std::vector<float> > features;
	changeStructure(plainDesc, features, descriptors.cols);
	
	BowVector vec;
	transform(features, vec);	
	return  (std::shared_ptr<MatchResultsBase>)query(vec, vt_params->amountToReturn);
}



std::shared_ptr<MatchResultsBase> ADTTreeIndex::search( const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example) 
{
	const std::shared_ptr<const SearchParams> &vt_params = (!params) ?std::make_shared<const SearchParams>(): std::static_pointer_cast<const SearchParams>(params);
	std::shared_ptr<ADTTreeIndex::MatchResults> match_result = std::make_shared<MatchResults>();

	// get descriptors for example
	if (!example) 
		return std::shared_ptr<MatchResultsBase>();

	const std::string &descriptors_location = datasetTest->feat_location(example->feature_path("descriptors"));
  
	if (!adtfilesystem::file_exists(descriptors_location))
		return std::shared_ptr<MatchResultsBase>();

	cv::Mat descriptors, descriptorsf;
	if (!adtfilesystem::load_cvmat(descriptors_location, descriptors)) 
		return std::shared_ptr<MatchResultsBase>();

	std::unordered_set<uint32_t> possibleMatches;
	descriptors.convertTo(descriptorsf, CV_32FC1);

	std::vector<float> vec = generateVector(descriptorsf, true, false, true, possibleMatches);
	typedef std::pair<uint64_t, float> matchPair;
	std::unordered_set<uint64_t> possibleImages;

	/// Problem: this indexes by the score so it will sort by the score, but if there are 2 equal scores this will dump one of the indexes. 
	/// Hopefully shouldn't matter
	std::map<float, uint32_t> scored_leaves;
	for (std::unordered_set<uint32_t>::iterator it = possibleMatches.begin(); it != possibleMatches.end(); it++)
	{
		uint32_t index = *it;
		float value = vec[numberOfNodes - invertedFiles.size() + index];
		scored_leaves[value] = index;
	}

	int imAdded = 0;
	for (std::map<float, uint32_t>::reverse_iterator it = scored_leaves.rbegin(); it != scored_leaves.rend() && imAdded < vt_params->cutoff; it++) 
	{
    
		std::unordered_map<uint64_t, uint32_t> & invFile = invertedFiles[it->second];

		typedef std::unordered_map<uint64_t, uint32_t>::iterator it_type;
		for (it_type iterator = invFile.begin(); iterator != invFile.end() && (imAdded++) < vt_params->cutoff; iterator++)
			if (possibleImages.count(iterator->first) == 0)
				possibleImages.insert(iterator->first);
	}


	std::vector<matchPair> values(possibleImages.size());;

	std::vector<uint64_t> possImagesVec(possibleImages.size());

	// push id's into vector
	int asdf = 0;
	for (std::unordered_set<uint64_t>::iterator it = possibleImages.begin(); it != possibleImages.end(); it++) 
	{
		possImagesVec[asdf++] = *it;
	}

	for (uint32_t i = 0; i < possImagesVec.size(); i++) 
	{
		uint64_t imID = possImagesVec[i];
		float score = 0;

		// load datavec from disk
		std::shared_ptr<ADTImage> image = std::static_pointer_cast<ADTImage>(datasetTest->image(imID));

		const std::vector<float> &dbVec = datasetTest->load_vec_feature(imID);

		for (uint32_t j = 0; j < numberOfNodes; j++) 
		{
			float t = vec[j] - dbVec[j];
			score += t*t;
		}

		values[i] = matchPair(imID, sqrt(score));
	}

	std::sort(values.begin(), values.end(), boost::bind(&std::pair<uint64_t, float>::second, _1) <boost::bind(&std::pair<uint64_t, float>::second, _2));
  
	if (values.size() > vt_params->amountToReturn) 
	{
		std::vector<matchPair>::iterator it = values.begin();
		values.erase(it + vt_params->amountToReturn, values.end());
	}

	for (uint32_t i = 0; i < values.size(); i++) 
	{
		match_result->matches_ids.push_back(values[i].first);
		match_result->tfidf_scores.push_back(values[i].second);
	}

	return (std::shared_ptr<MatchResultsBase>)match_result;
}
  
uint32_t ADTTreeIndex::tree_splits() const 
{
	return tiParams->split;
}

uint32_t ADTTreeIndex::tree_depth() const 
{
	return tiParams->depth;
}
