#pragma once
/*satish.lokkoju@gmail.com*/
/*22/08/2015*/
/*Simple class to hold Tree index related class*/
/*etc.*/
#include "ADTSearchBase.hpp"
#include "ADTBOWmodel.hpp"
#include <unordered_map>
#include <unordered_set>

class ADTTreeIndex : public ADTSearchBase 
{
public:

	/// Subclass of train params base which specifies vocab tree training parameters.
	struct TITrainParams : public TrainParamsBase 
	{
		uint32_t depth; // tree depth
		uint32_t split; // number of children per node
		WeightingType wt;
		LNorm lnorm;
		ScoringType st;
		int numberofTrainingImgs;
		std::string descType;
	};

	/// Subclass of train params base which specifies Vocab Tree training parameters.
	struct SearchParams : public SearchParamsBase 
	{
		SearchParams(uint64_t cutoff = 4096) : cutoff(cutoff) 
		{ 
		}
    
		uint32_t amountToReturn;
		uint32_t cutoff;
	};

	/// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase 
	{
		std::vector<float> tfidf_scores;
		double ukbenchScore;
	};

	ADTTreeIndex(std::shared_ptr<TITrainParams> tip);

	~ADTTreeIndex()
	{
		if(NULL !=m_scoring_object)
			delete m_scoring_object;
	}

	/// Given a set of training parameters, trains.  Returns true if successful, false if not successful.
	bool train(const std::shared_ptr<ADTDataset> &dataset);
	bool train2(const std::shared_ptr<ADTDataset>  &dataset);

	/// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);
	void load2(const std::string &file_name);
	bool load_db(const std::string &file_name);

	bool init_db_memory(const std::string &file_name);

	/// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;
	bool save2(const std::string &filename) const;
	bool save_db(const std::string &filename) const;
	bool save_db_woid(const std::string &filename,int image_id) const;

	/// Given a set of search parameters, a query image, searches for matching images and returns the match
	std::shared_ptr<MatchResultsBase> search(const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example);
	std::shared_ptr<MatchResultsBase> search2(const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example); 
	std::shared_ptr<MatchResultsBase> search_user( const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<ADTDataset> &datasetTest,const std::shared_ptr<const ADTImage > &example);
	/// returns the split size of each node
	uint32_t tree_splits() const;
	/// returns the depth size of tree
	uint32_t tree_depth() const;
	void addImagetoUserDatabase(const std::shared_ptr<ADTDataset> &dataset,std::shared_ptr<const ADTImage > image);
	bool addtoDatabase(const std::shared_ptr<ADTDataset>  &dataset,const std::vector< std::shared_ptr<const ADTImage > > &examples);

protected:	  
	std::shared_ptr<TITrainParams> tiParams;
	/* Inverted file declaration */
  
	/// Item of IFRow
	struct IFPair
	{
		/// Entry id
		uint32_t entry_id;
    
		/// Word weight in this entry
		double word_weight;
    
		/**
		* Creates an empty pair
		*/

		IFPair()
		{

		}
    
		/**
		* Creates an inverted file pair
		* @param eid entry id
		* @param wv word weight
		*/

		IFPair(uint32_t eid, double wv): entry_id(eid), word_weight(wv) 
		{

		}
    
		/**
		* Compares the entry ids
		* @param eid
		* @return true iff this entry id is the same as eid
		*/
		inline bool operator==(uint32_t eid) const 
		{
			return entry_id == eid; 
		}
	};
  
	/// Row of InvertedFile
	typedef std::list<IFPair> IFRow;
	// IFRows are sorted in ascending entry_id order
  
	/// Inverted index
	typedef std::vector<IFRow> InvertedFile; 
	// InvertedFile[word_id] --> inverted file of that word
	struct TreeNode 
	{
		uint32_t invertedFileLength;
		
		/// Node id
		uint32_t id;
		/// Weight if the node is a word
		double weight;
		/// Children 
		std::vector<uint32_t> children;
		/// Parent node (undefined in case of root)
		uint32_t parent;
		/// Node descriptor
		std::vector<float> descriptor;

		/// range from 0..maxLevel-1
		uint32_t level; 
		/// index for this level, ranging from 0..split^level-1
		/// For example: the first level children will simply have indexes from 0..split-1
		///   for the second level the children of the first child will have 0..split-1
		///   while the children of the second child we have split..2*split-1
		/// This will be used to identify the node and used to index into the vectors for images
		uint32_t levelIndex; 

		/// index in a level order traversal of the tree
		uint32_t index;
		// node Mean
		cv::Mat mean;
		/// index into the array of nodes of the first child, all children are next to eachother
		/// if this is = 0 then it is a leaf (because the root can never be a child)
		uint32_t firstChildIndex;

		uint32_t word_id;
		/**
		 * Empty constructor
		 */
		TreeNode(): id(0), weight(0), parent(0), word_id(0){}
    
		/**
		 * Constructor
		 * @param _id node id
		 */
		TreeNode(uint32_t _id): id(_id), weight(0), parent(0), word_id(0)
		{

		}

		/**
		 * Returns whether the node is a leaf node
		 * @return true iff the node is a leaf
		 */
		inline bool isLeaf() const 
		{
			return children.empty(); 
		}
	};
	/// Object for computing scores
	GeneralScoring* m_scoring_object;
  
	/// Tree nodes
	std::vector<TreeNode> m_nodes;
  
	/// Words of the vocabulary (tree leaves)
	/// this condition holds: m_words[wid]->word_id == wid
	std::vector<TreeNode*> m_words;

	/// number of nodes the tree will have, saved in variable so don't have to recompute
	uint32_t numberOfNodes;

	std::vector<float> weights;
	std::vector<TreeNode> tree;
	std::vector<std::unordered_map<uint64_t, uint32_t>> invertedFiles;

	/// Stores the database vectors for all images in the database - d_i in the paper
	/// Indexes by the image id
	/// std::unordered_map<uint64_t, std::vector<float>> databaseVectors;

	/// Recursively builds a tree, starting with 0 and ending with currLevel = maxLevel-1
	/// The arguments of indices, maxNode, and ratio are only used for multinode mpi. In this case descriptors will always contain all the descriptors
	/// and indices will index into it. Nodes will be able to send additional work to the nodes [rank:maxNode], so if maxNode=rank then can't 
	/// delegate any more work to processors. Ratio is the ideal number of descriptors per processor, computed before hand and carried down
	void buildTreeRecursive(uint32_t t, const cv::Mat &descriptors, cv::TermCriteria &tc, int attempts, int flags, int currLevel);

	/// helper function, inserts a dummy possibleMatches
	std::vector<float> generateVector(const cv::Mat &descriptors, bool shouldWeight, bool building, bool multinode, int64_t id = -1);

	/// To call with an id call without possibleMatches and it will go to the helper function
	/// Takes descriptors for an image and for each descriptor finds the path down the tree generating a vector (describing the path)
	/// Adds up all vectors (one from each descriptor) to return the vector of counts for each node
	/// If  shouldWeight is true will weight each by the weight of the node, should be true for general query and false for construction
	/// If id is set then will insert that id into the invertedFile of each leaf visited, if negative or not set then won't do anything
	/// When building is false will use insert images into possibleMatches, possibleMatches will not be used if building is false
	/// If multinode is true then all the descriptors will be run over multiple nodes
	std::vector<float> generateVector(const cv::Mat &descriptors, bool shouldWeight, bool building, bool multinode, std::unordered_set<uint32_t> & possibleMatches, int64_t id = -1);

	/// Recursive function that recursively goes down the tree from t to find where the single descriptor belongs (stopping at leaf)
	/// On each node increments cound in the counts vector
	/// If id is set (>=0) then adds the image with that id to the leaf
	/// Picks the child to traverse down based on the max dot product
	void generateVectorHelper(uint32_t nodeIndex, const cv::Mat &descriptor, std::vector<float> & counts, std::unordered_set<uint32_t> & possibleMatches, bool building, int64_t id = -1);
private:
	void loadFeatures(const std::shared_ptr<ADTDataset>  &dataset,const std::vector< std::shared_ptr<const ADTImage > > &trainingImages,std::vector<std::vector<std::vector<float> > > &features);
	void changeStructure(const std::vector<float> &plain, std::vector<std::vector<float> > &out, int L);
	void getFeatures( const std::vector<std::vector<std::vector<float>> > &training_features, std::vector<const std::vector<float> *> &featuresp) const;
	void setNodeWeights(const std::vector<std::vector<std::vector<float>> > &training_features);
	void transform(const std::vector<float> &feature,uint32_t &word_id, double &weight, uint32_t *nid = NULL, int levelsup = 0) const;
	void createWords();
	void HKmeansStep(uint32_t parent_id, const std::vector<const std::vector<float> *> &descriptors, int current_level);
	void initiateClusters(const std::vector<const std::vector<float> *> &descriptors, std::vector<std::vector<float> > &clusters) const;
	void initiateClustersKMpp(const std::vector<const std::vector<float> *> &pfeatures, std::vector<std::vector<float> > &clusters) const;
	void save(cv::FileStorage &f,const std::string &name = "vocabulary") const;
	bool save_db(cv::FileStorage &f,const std::string &name = "index") const;
	bool load_db(const cv::FileStorage &fs,const std::string &name = "index");
	void transform(const std::vector<std::vector<float>>& features, BowVector &v) const;
	uint32_t add(const BowVector &v,uint64_t id);
    /// Number of valid entries in m_dfile
    int m_nentries;

	  /// Inverted file (must have size() == |words|)
	InvertedFile m_invfile;
	void addImagetoDatabase(std::vector<std::vector<float> > &featureV,uint64_t id);
	std::shared_ptr<MatchResultsBase> query(const BowVector &vec, int max_results) const;
	uint32_t descriptorSize;
	//std::shared_ptr<ADTDataset> datasetStored;
};
