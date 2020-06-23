#pragma once

/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Class to hold all the wrapper functions for image retrieval systems*/
/*etc.*/
/*Modifications:22/08/2015*/
/*added support for Vocabulary Tree and improved the search and database classes design*/

//#include "ADTBuilddb.hpp"
//#include "ADTMatch.hpp"
#include "retrieval_lib/ADTDataset.hpp"
#include "retrieval_lib/ADTDefines.hpp"
#include "retrieval_lib/ADTBOWmodel.hpp"
#include "retrieval_lib/ADTInvertedIndex.hpp"
#include "retrieval_lib/ADTmatchpages.hpp"
#include "retrieval_lib/ADTfilesystem.hpp"
#include "retrieval_lib/ADTVision.hpp"
#include "retrieval_lib/ADTTreeIndex.hpp"
#include "retrieval_lib/ADTCycleTimer.hpp"
#include "ADTUserManagement.hpp"

#include <stdint.h>

class ADTRetrieval
{
public:
	ADTRetrieval(int numbimagesfortraining,RetrievalType type,std::string descType)
	{
		uint32_t bow_clusters[] = { 256, 3125, 46656 };

		std::shared_ptr<ADTTreeIndex::TITrainParams>tiparams = std::make_shared<ADTTreeIndex::TITrainParams>();
		tiparams->depth =3;
		tiparams->descType =descType;
		tiparams->split = 4;
		tiparams->wt =TF_IDF;
		tiparams->st = L1_NORM;
		tiparams->numberofTrainingImgs = numbimagesfortraining;
			
		std::shared_ptr<ADTBagofWords::TrainParams> bwparams = std::make_shared<ADTBagofWords::TrainParams>();
		bwparams->descType = descType;
		bwparams->numberofTrainingImgs = numbimagesfortraining;
		bwparams->numClusters = bow_clusters[0];
		ti  = std::make_shared<ADTTreeIndex>(tiparams);
		ii  = std::make_shared<ADTInvertedIndex>();
		bow = std::make_shared<ADTBagofWords>(bwparams);
		retrievalMethod = type;
	};

	ADTRetrieval(int numbimagesfortraining,RetrievalType type,std::string descType,std::string users_folder)
	{
		uint32_t bow_clusters[] = { 256, 3125, 46656 };

		std::shared_ptr<ADTTreeIndex::TITrainParams>tiparams = std::make_shared<ADTTreeIndex::TITrainParams>();
		tiparams->depth =4;
		tiparams->descType =descType;
		tiparams->split = 6;
		tiparams->wt =TF_IDF;
		tiparams->st = L1_NORM;
		tiparams->numberofTrainingImgs = numbimagesfortraining;
			
		std::shared_ptr<ADTBagofWords::TrainParams> bwparams = std::make_shared<ADTBagofWords::TrainParams>();
		bwparams->descType = descType;
		bwparams->numberofTrainingImgs = numbimagesfortraining;
		bwparams->numClusters = bow_clusters[0];
		ti  = std::make_shared<ADTTreeIndex>(tiparams);
		ii = std::make_shared<ADTInvertedIndex>();
		bow = std::make_shared<ADTBagofWords>(bwparams);
		userMgt = std::make_shared<ADTUserManagement>(users_folder);
		retrievalMethod = type;
	};
	
	ADTRetrieval(RetrievalType type,std::string descType,std::string modelPath,std::string users_folder)
	{
		std::shared_ptr<ADTTreeIndex::TITrainParams>tiparams = std::make_shared<ADTTreeIndex::TITrainParams>();
		tiparams->depth =4;
		tiparams->descType =descType;
		tiparams->split = 6;
		tiparams->wt =TF_IDF;
		tiparams->st = L1_NORM;
		tiparams->numberofTrainingImgs = 1000;

		ti  = std::make_shared<ADTTreeIndex>(tiparams);
		loadmodel(modelPath);
		userMgt = std::make_shared<ADTUserManagement>(users_folder);
		retrievalMethod = type;
	};

	~ADTRetrieval()
	{

	};

	bool learn(const std::shared_ptr<ADTDataset> &dataset);
	bool loadmodel(const std::string &pathofmodel);

	bool addUser(int userid,const std::string databaseLocat);
	bool addUser(int userid);
	bool addUser(std::string emailid);

	int64_t addImageUser(int userid,const std::string pathofImage);
	int64_t addImageUser(std::string emailid,const std::string pathofImage);
	bool removeImageUser(std::string emailid,std::string imagename);

	std::vector<int>  retrieveImage(int userid, cv::Mat image,int topn);
	std::vector<std::string>  retrieveImage(std::string emailid, cv::Mat image,int topn);
	bool deletUser(int userid);
	bool deletUser(std::string emailid);

#ifdef EVALUATE
	bool addtoDatabase(const std::shared_ptr<ADTDataset> &dataset);
	void searchUserDB_testImages(int userid,const std::string usersLocation);
	void searchUserDB_testImages(std::string emailid,const std::string usersLocation);
	bool testRandomImages(const std::shared_ptr<ADTDataset> &datasetTest,std::vector<std::shared_ptr<const ADTImage> > &random_images);
#endif

private:
	void train_bow(const std::shared_ptr<ADTDataset>  &dataset);
	double getScore(std::shared_ptr<const SimpleDataset::SimpleImage> query_image ,std::shared_ptr<MatchResultsBase> matchresults);
	
	RetrievalType retrievalMethod;
	std::shared_ptr<ADTInvertedIndex> ii;
	std::shared_ptr<ADTTreeIndex> ti;
	std::shared_ptr<ADTBagofWords> bow;
	std::shared_ptr<ADTUserManagement> userMgt;
};
