#include "ADTRetrieval.hpp"


bool ADTRetrieval::learn(const std::shared_ptr<ADTDataset> &datasetTrain)
{
	if(retrievalMethod == INVERTEDINDEX)
	{
		INFO("Training bag of words");
		train_bow(datasetTrain);
		INFO("Computing bag of words features");
		bow->compute_bow_features(datasetTrain);
	}
	else
	{
		std::stringstream vocab_output_file;
		vocab_output_file << datasetTrain->location() << "/tree/" << "vocab_"<<ti->tree_depth()<<"_"<<ti->tree_splits()<< ".tree";

		if(!adtfilesystem::file_exists(vocab_output_file.str()))
		{
			INFO("Training Vocabulary Tree");
			ti->train2(datasetTrain);
			INFO("Vocabulary Training complete");
			INFO("Saving vocabulary...");
			ti->save2(vocab_output_file.str());
		}
		else
		{
			INFO("Loading Vocabulary Tree");
			ti->load2(vocab_output_file.str());
		}
	}

	return true;
}

bool ADTRetrieval::loadmodel(const std::string &modellocation)
{

	std::stringstream vocab_output_file;
	vocab_output_file << modellocation << "/tree/" << "vocab_"<<ti->tree_depth()<<"_"<<ti->tree_splits()<< ".tree";

	if(!adtfilesystem::file_exists(vocab_output_file.str()))
	{
		INFO("Model Doesnt exits!");
	}
	else
	{
		INFO("Loading Vocabulary Tree");
		ti->load2(vocab_output_file.str());
	}

	return true;
}

#ifdef EVALUATE
std::string query_image_filename;
bool isEqual(const std::pair<std::string, std::vector<std::string>> element)
{
	return element.first ==  query_image_filename;
}

#endif
double ADTRetrieval::getScore(std::shared_ptr<const SimpleDataset::SimpleImage> query_image ,std::shared_ptr<MatchResultsBase> matchresults)
{
#if 1
	int query_id = query_image->id;
	int startIndex = ((query_id)/4)*4;
	int results[4] = {startIndex,startIndex+1,startIndex+2,startIndex+3};
	
	double score =0;
	
	for(int i =0; i< 4; i++)
	{
		int id = matchresults->matches_ids[i];

		if((id==results[0])||(id==results[1])||(id==results[2])||(id==results[3]))
		{
			score++;
		}
	}

	return score;
#else
	cv::FileStorage fs_in("ground_truth.yml.gz", cv::FileStorage::READ);
	std::vector< std::pair<std::string, std::vector<std::string>> > sortList;
	std::vector< std::pair<std::string,  std::vector<std::string>> >::iterator it_sortList;

	cv::FileNode n = fs_in["test_images"];
	if (n.type() != cv::FileNode::SEQ)
	{
		std::cerr << "images is not a sequence! FAIL" << std::endl;
		return 1;
	}
	cv::FileNodeIterator it = n.begin(), it_end = n.end();
	int i=0;
	for (; it != it_end; ++it)
	{
		std::vector<std::string> resuts_vec;
		std::string query =  (std::string)(*it)["query"];
		cv::FileNode results = (*it)["results"];
		cv::FileNodeIterator it_results = results.begin(), it_results_end = results.end();
		for (; it_results != it_results_end; ++it_results)
		{
			resuts_vec.push_back( (std::string)(*it_results));
		}
		
		sortList.push_back(std::make_pair(query,resuts_vec));
		i++;
	}
	fs_in.release();
	query_image_filename = query_image->get_filename();

	it_sortList = std::find_if( sortList.begin(), sortList.end(), isEqual );
	std::vector<std::string> results_final = (*it_sortList).second;

#endif
}

void ADTRetrieval::searchUserDB_testImages(std::string emailid,const std::string usersLocation)
{
	std::stringstream ss;
	std::string currentUserLoc =  userMgt->getUserLocation(emailid);
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> testdatasetUser = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	uint32_t num_validate = 10;
	
	std::vector<std::shared_ptr<const ADTImage> > random_images = testdatasetUser->random_images(100);
	uint32_t numberofImages_test = random_images.size();

	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "Email ID :";
		ss << emailid;
		INFO(ss.str());
		INFO("ERROR:Images of this user are not indexed");
		exit(1);
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}

   if(retrievalMethod == INVERTEDINDEX)
   {
		INFO("INVERTEDINDEX is not supported for Restricted Search");
		exit(1);
   }
   else
   {
		// search index
		INFO("Running Tree search");
		
		ADTMatchesPage html_output_index;
		uint32_t	total_correct = 0, total_tested = 0;

		std::shared_ptr<ADTTreeIndex::SearchParams> vt_params =  std::make_shared<ADTTreeIndex::SearchParams>();
		vt_params->amountToReturn = 64;
		double Score = 0;
		double totalTime =0;
		for(uint32_t i = 0; i < numberofImages_test; i++)
		{
			double start_time_query = CycleTimer::currentSeconds();
			std::shared_ptr<const SimpleDataset::SimpleImage> query_image = std::static_pointer_cast <const SimpleDataset::SimpleImage > (random_images[i]);
			INFO("Searching for " +query_image->get_location());
			std::shared_ptr<ADTTreeIndex::MatchResults>	matches_index = std::static_pointer_cast <ADTTreeIndex::MatchResults >	(ti->search2(vt_params,testdatasetUser, query_image));

			if(matches_index == std::shared_ptr < ADTTreeIndex::MatchResults > ())
			{
				INFO("Error while running search.");
				continue;
			}
			
			double end_time_query = CycleTimer::currentSeconds();
			
			std::stringstream ss;
			ss.str("");
			ss << "Query Processing time is :";
			ss << (end_time_query- start_time_query)*1000;
			ss << " milli seconds";
			INFO(ss.str());
			totalTime += end_time_query- start_time_query;

#ifdef EVALUATE
			Score += getScore(query_image,matches_index);
			// validate matches
			cv::Mat keypoints_0, descriptors_0;
			const std::string & query_keypoints_location = testdatasetUser->feat_location(query_image->feature_path("keypoints"));
			const std::string & query_descriptors_location = testdatasetUser->feat_location(query_image->feature_path("descriptors"));
			adtfilesystem::load_cvmat(query_keypoints_location, keypoints_0);
			adtfilesystem::load_cvmat(query_descriptors_location, descriptors_0);
			std::vector<int> validated(MIN(num_validate, matches_index->matches_ids.size()), 0);
			total_tested += validated.size();

			uint32_t	total_correct_tmp = 0;

			for(uint32_t j = 0; j < validated.size(); j++)
			{
				cv::Mat keypoints_1, descriptors_1;
				std::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(testdatasetUser->image(matches_index->matches_ids[j]));
			
				const std::string & match_keypoints_location = testdatasetUser->feat_location(match_image->feature_path("keypoints"));
				const std::string & match_descriptors_location = testdatasetUser->feat_location(match_image->feature_path("descriptors"));
			
				adtfilesystem::load_cvmat(match_keypoints_location, keypoints_1);
				adtfilesystem::load_cvmat(match_descriptors_location, descriptors_1);

				cv::detail::MatchesInfo match_info;
				ADTVision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

				validated[j] = ADTVision::is_good_match(match_info) ? 1 : -1;
			
				if(validated[j] > 0) 
					total_correct_tmp++;
			}

			total_correct += total_correct_tmp;
			html_output_index.add_match(query_image,matches_index->matches_ids,testdatasetUser,std::make_shared < std::vector<int> > (validated));

			std::stringstream	outfilestr;
			outfilestr << testdatasetUser->location() << "/results/matches/tree_" << ti->tree_splits()<<"_"<<ti->tree_depth();
			html_output_index.write(outfilestr.str());
#endif
		}

		std::stringstream ss;

#ifdef EVALUATE
		ss.str("");
		ss << "UKbench Score is: ";
		ss << Score/numberofImages_test;
		INFO(ss.str());
#endif

		ss.str("");
		ss << "Average query Processing time is: ";
		ss << 1000*totalTime/numberofImages_test;
		ss << " milli seconds";
		INFO(ss.str());
   }
}

void ADTRetrieval::searchUserDB_testImages(int userid,const std::string usersLocation)
{
	std::stringstream ss;
	ss.str("");
	ss << "/user_";
	ss << userid;
	std::string currentUserLoc = usersLocation+ss.str();
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> testdatasetUser = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	uint32_t num_validate = 10;
	
	std::vector<std::shared_ptr<const ADTImage> > random_images = testdatasetUser->random_images(100);
	uint32_t numberofImages_test = random_images.size();

	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "User ID :";
		ss << userid;
		INFO(ss.str());
		INFO("ERROR:Images of this user are not indexed");
		exit(1);
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}

   if(retrievalMethod == INVERTEDINDEX)
   {
		INFO("INVERTEDINDEX is not supported for Restricted Search");
		exit(1);
   }
   else
   {
		// search index
		INFO("Running Tree search");
		
		ADTMatchesPage html_output_index;
		uint32_t	total_correct = 0, total_tested = 0;

		std::shared_ptr<ADTTreeIndex::SearchParams> vt_params =  std::make_shared<ADTTreeIndex::SearchParams>();
		vt_params->amountToReturn = 64;
		double Score = 0;
		double totalTime =0;
		for(uint32_t i = 0; i < numberofImages_test; i++)
		{
			double start_time_query = CycleTimer::currentSeconds();
			std::shared_ptr<const SimpleDataset::SimpleImage> query_image = std::static_pointer_cast <const SimpleDataset::SimpleImage > (random_images[i]);
			INFO("Searching for " +query_image->get_location());
			std::shared_ptr<ADTTreeIndex::MatchResults>	matches_index = std::static_pointer_cast <ADTTreeIndex::MatchResults >	(ti->search2(vt_params,testdatasetUser, query_image));

			if(matches_index == std::shared_ptr < ADTTreeIndex::MatchResults > ())
			{
				INFO("Error while running search.");
				continue;
			}
			
			double end_time_query = CycleTimer::currentSeconds();
			
			std::stringstream ss;
			ss.str("");
			ss << "Query Processing time is :";
			ss << (end_time_query- start_time_query)*1000;
			ss << " milli seconds";
			INFO(ss.str());
			totalTime += end_time_query- start_time_query;

#ifdef EVALUATE
			Score += getScore(query_image,matches_index);
			// validate matches
			cv::Mat keypoints_0, descriptors_0;
			const std::string & query_keypoints_location = testdatasetUser->feat_location(query_image->feature_path("keypoints"));
			const std::string & query_descriptors_location = testdatasetUser->feat_location(query_image->feature_path("descriptors"));
			adtfilesystem::load_cvmat(query_keypoints_location, keypoints_0);
			adtfilesystem::load_cvmat(query_descriptors_location, descriptors_0);
			std::vector<int> validated(MIN(num_validate, matches_index->matches_ids.size()), 0);
			total_tested += validated.size();

			uint32_t	total_correct_tmp = 0;

			for(uint32_t j = 0; j < validated.size(); j++)
			{
				cv::Mat keypoints_1, descriptors_1;
				std::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(testdatasetUser->image(matches_index->matches_ids[j]));
			
				const std::string & match_keypoints_location = testdatasetUser->feat_location(match_image->feature_path("keypoints"));
				const std::string & match_descriptors_location = testdatasetUser->feat_location(match_image->feature_path("descriptors"));
			
				adtfilesystem::load_cvmat(match_keypoints_location, keypoints_1);
				adtfilesystem::load_cvmat(match_descriptors_location, descriptors_1);

				cv::detail::MatchesInfo match_info;
				ADTVision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

				validated[j] = ADTVision::is_good_match(match_info) ? 1 : -1;
			
				if(validated[j] > 0) 
					total_correct_tmp++;
			}

			total_correct += total_correct_tmp;
			html_output_index.add_match(query_image,matches_index->matches_ids,testdatasetUser,std::make_shared < std::vector<int> > (validated));

			std::stringstream	outfilestr;
			outfilestr << testdatasetUser->location() << "/results/matches/tree_" << ti->tree_splits()<<"_"<<ti->tree_depth();
			html_output_index.write(outfilestr.str());
#endif
		}

		std::stringstream ss;

#ifdef EVALUATE
		ss.str("");
		ss << "UKbench Score is: ";
		ss << Score/numberofImages_test;
		INFO(ss.str());
#endif

		ss.str("");
		ss << "Average query Processing time is: ";
		ss << 1000*totalTime/numberofImages_test;
		ss << " milli seconds";
		INFO(ss.str());
   }
}

bool ADTRetrieval::addUser(int userid,const std::string usersLocation)
{
	std::stringstream ss;
	ss.str("");
	std::string currentUserLoc = usersLocation;
	if(!adtfilesystem::is_directory(currentUserLoc))
	{
		INFO("ERROR: the data related to user doesnt exits");
		ss.str("");
		ss << currentUserLoc;
		ss << " ... doesn't Exist";
		INFO(ss.str());
		return false;
	}
	
	ss.str("");
	ss << currentUserLoc;
	ss << "/user.db";
	std::string userDBname = ss.str();
	std::shared_ptr<SimpleDataset> datasetUser = std::make_shared<SimpleDataset>(currentUserLoc,userDBname,"AKAZE");
	
	ss.str("");
	ss << "INFO: Adding Image to User id :";
	ss << userid <<std::endl;
	ss << "INFO: User location :";
	ss << currentUserLoc << std::endl;
	ss << "INFO: database location :";
	ss << userDBname << std::endl;

	INFO(ss.str());
	if(INVERTEDINDEX == retrievalMethod)
	{
		INFO("INVERTEDINDEX is not supported for Restricted Search");
		exit(1);
	}
	else
	{
		std::stringstream indexed_database_location("");
		indexed_database_location << datasetUser->location() << "/indexed/" << "index_"<< ".invertedIndex";

		if(!adtfilesystem::file_exists(indexed_database_location.str()))
		{
			INFO("Adding images to the database: Vocabulary Index");
			ti->addtoDatabase(datasetUser,datasetUser->all_images());
			ti->save_db(indexed_database_location.str());
		}
	}

	return userMgt->addUser(userid);
}

int64_t  ADTRetrieval::addImageUser(int userid,const std::string pathofImage)
{	
	std::stringstream ss;
	std::string currentUserLoc = userMgt->getUserLocation(userid);

	if(currentUserLoc.empty())
	{
		ss.str("");
		ss << "User ";
		ss << userid;
		ss <<" doesn't exist !";
		INFO(ss.str());
		return -1;
	}
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> userdataset = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	int64_t image_id = userdataset->add_image(pathofImage);

	if(image_id <0)
	 return image_id;

	userdataset->write(ss.str());
		
	ss.str("");
	ss << "Adding the image " << pathofImage << " to user with user id " << userid;
	INFO(ss.str());
	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "User ID :";
		ss << userid;
		INFO(ss.str());
		INFO("First Image being added to the user");
		ti->init_db_memory(indexed_database_location.str());
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}
	std::shared_ptr<const SimpleDataset::SimpleImage> simage = std::make_shared<const SimpleDataset::SimpleImage>(pathofImage, image_id);
	ti->addImagetoUserDatabase(userdataset,simage);
	ti->save_db(indexed_database_location.str());
	return image_id;
}

bool  ADTRetrieval::addUser(std::string emailid)
{
	return userMgt->addUser(emailid);
}

int64_t  ADTRetrieval::addImageUser(std::string emailid,const std::string pathofImage)
{
	std::stringstream ss;
	std::string currentUserLoc = userMgt->getUserLocation(emailid);

	if(currentUserLoc.empty())
	{
		ss.str("");
		ss << "User ";
		ss << emailid;
		ss <<" doesn't exist !";
		INFO(ss.str());
		return -1;
	}
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> userdataset = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	int64_t image_id = userdataset->add_image(pathofImage);
	if(image_id <0)
	 return image_id;
	userdataset->write(ss.str());
	
	ss.str("");
	ss << "Adding the image " << pathofImage << " to user with email id " << emailid;
	INFO(ss.str());
	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "User ID :";
		ss << emailid;
		INFO(ss.str());
		INFO("First Image being added to the user");
		ti->init_db_memory(indexed_database_location.str());
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}
	
	std::shared_ptr<const SimpleDataset::SimpleImage> simage = std::make_shared<const SimpleDataset::SimpleImage>(pathofImage, image_id);
	ti->addImagetoUserDatabase(userdataset,simage);
	ti->save_db(indexed_database_location.str());

	return image_id;
}

bool  ADTRetrieval::removeImageUser(std::string emailid,std::string imagename)
{
	std::stringstream ss;
	std::string currentUserLoc = userMgt->getUserLocation(emailid);
		
	if(currentUserLoc.empty())
	{
		ss.str("");
		ss << "User ";
		ss << emailid;
		ss <<" doesn't exist !";
		INFO(ss.str());
		return -1;
	}
	
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> userdataset = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";
	int64_t imageid = userdataset->delete_image(imagename);

	if(imageid <0)
	  return false;

	userdataset->write(ss.str());

	ss.str("");
	ss << "Removing the Image :" << imagename << " from user with email id :"  << emailid;
	INFO(ss.str());

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "User ID :";
		ss << emailid;
		INFO(ss.str());
		INFO("ERROR:Images of this user are not indexed");
		exit(1);
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}
	
	ti->save_db_woid(indexed_database_location.str(),imageid);

	return true;
}


std::vector<std::string>   ADTRetrieval::retrieveImage(std::string emailid, cv::Mat query_image_mat,int topn)
{
	std::vector<std::string> output;
	std::stringstream ss;
	std::string currentUserLoc = userMgt->getUserLocation(emailid);

	if(currentUserLoc.empty())
	{
		ss.str("");
		ss << "User ";
		ss << emailid;
		ss <<" doesn't exist !";
		INFO(ss.str());
		output.resize(0);
		return output;
	}
	
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> userdataset = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "User ID :";
		ss << emailid;
		INFO(ss.str());
		INFO("ERROR:Images of this user are not indexed");
		output.resize(0);
		return output;
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}
	
	ss.str("");
	ss << currentUserLoc;
	ss << "/search/temp.jpg";
	adtfilesystem::create_file_directory(ss.str());
	
	if(!query_image_mat.empty())
	{
	     cv::imwrite(ss.str(),query_image_mat);
	}
	else
	{
	     INFO("Query Image is empty !");
	     return output;
	}
	
	std::shared_ptr<const SimpleDataset::SimpleImage> query_image = std::make_shared<const SimpleDataset::SimpleImage>(ss.str(), 100);
	if(retrievalMethod == INVERTEDINDEX)
	{
		INFO("INVERTEDINDEX is not supported for Restricted Search");
		exit(1);
	}
	else
	{
		// search index
		INFO("Running Tree search");
		std::shared_ptr<ADTTreeIndex::SearchParams> vt_params =  std::make_shared<ADTTreeIndex::SearchParams>();
		vt_params->amountToReturn = 64;
		double Score = 0;
		double totalTime =0;

		double start_time_query = CycleTimer::currentSeconds();
		INFO("Searching for " + query_image->get_location());
		std::shared_ptr<ADTTreeIndex::MatchResults>	matches_index = std::static_pointer_cast <ADTTreeIndex::MatchResults >	(ti->search_user(vt_params,userdataset, query_image));

		if(matches_index == std::shared_ptr < ADTTreeIndex::MatchResults > ())
		{
			INFO("Error while running search.");
			return output;
		}

		uint32_t num_validate = 10;
		// validate matches
		cv::Mat im = cv::imread(query_image->get_location(),cv::IMREAD_GRAYSCALE);
		cv::Mat keypoints_0,descriptors_0;
		ADTVision::compute_sparse_akaze_feature(im,keypoints_0,descriptors_0);
		std::vector<int> validated(MIN(num_validate, matches_index->matches_ids.size()), 0);

		for(uint32_t j = 0; j < validated.size(); j++)
		{
			cv::Mat keypoints_1, descriptors_1;
			std::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(userdataset->image(matches_index->matches_ids[j]));
			
			const std::string & match_keypoints_location = userdataset->feat_location(match_image->feature_path("keypoints"));
			const std::string & match_descriptors_location = userdataset->feat_location(match_image->feature_path("descriptors"));
			
			adtfilesystem::load_cvmat(match_keypoints_location, keypoints_1);
			adtfilesystem::load_cvmat(match_descriptors_location, descriptors_1);

			cv::detail::MatchesInfo match_info;
			ADTVision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

			validated[j] = ADTVision::is_good_match(match_info) ? 1 : -1;
			
			if(validated[j] > 0)
			{
				output.push_back(match_image->get_filename());			
			}
		}

#ifdef EVALUATE
		ADTMatchesPage html_output_index;
		html_output_index.add_match(query_image,matches_index->matches_ids,userdataset,std::make_shared < std::vector<int> > (validated));

		std::stringstream	outfilestr;
		outfilestr << userdataset->location() << "/results/matches/tree_" << ti->tree_splits()<<"_"<<ti->tree_depth();
		html_output_index.write(outfilestr.str());

#endif

	}

	return output;
}

std::vector<int> ADTRetrieval::retrieveImage(int userid, cv::Mat query_image_mat,int topn)
{
	std::vector<int> output;
	output.resize(topn);
	std::stringstream ss;
	std::string currentUserLoc = userMgt->getUserLocation(userid);

	if(currentUserLoc.empty())
	{
		ss.str("");
		ss << "User ";
		ss << userid;
		ss <<" doesn't exist !";
		INFO(ss.str());
		output.resize(0);
		return output;
	}

	uint32_t num_validate = 10;
	
	ss.str("");
	ss << currentUserLoc;
	ss  << "/user.db";

	std::shared_ptr<SimpleDataset> userdataset = std::make_shared<SimpleDataset>(currentUserLoc,ss.str(),"AKAZE");
	ss.str("");
	ss << currentUserLoc;
	ss << "/search/temp.jpg";
	adtfilesystem::create_file_directory(ss.str());
	
	if(!query_image_mat.empty())
	{
		cv::imwrite(ss.str(),query_image_mat);
	}
	else
	{
		INFO("Query Image is empty !");
		return output;
	}
	
	std::shared_ptr<const SimpleDataset::SimpleImage> query_image = std::make_shared<const SimpleDataset::SimpleImage>(ss.str(), 100);
	// load the indexed database if exits
	std::stringstream indexed_database_location("");
	indexed_database_location << currentUserLoc << "/indexed/" << "index_"<< ".invertedIndex";

	if(!adtfilesystem::file_exists(indexed_database_location.str()))
	{
		ss.str("");
		ss << "User ID :";
		ss << userid;
		INFO(ss.str());
		INFO("ERROR:Images of this user are not indexed");
		exit(1);
	}
	else
	{
		ti->load_db(indexed_database_location.str());
	}

	if(retrievalMethod == INVERTEDINDEX)
	{
		INFO("INVERTEDINDEX is not supported for Restricted Search");
		exit(1);
	}
	else
	{
		// search index
		INFO("Running Tree search");
		uint32_t	total_correct = 0, total_tested = 0;

		std::shared_ptr<ADTTreeIndex::SearchParams> vt_params =  std::make_shared<ADTTreeIndex::SearchParams>();
		vt_params->amountToReturn = 64;
		double Score = 0;
		double totalTime =0;
		
		double start_time_query = CycleTimer::currentSeconds();
		INFO("Searching for " + query_image->get_location());
		std::shared_ptr<ADTTreeIndex::MatchResults> matches_index = std::static_pointer_cast <ADTTreeIndex::MatchResults>(ti->search_user(vt_params,userdataset, query_image));

		if(matches_index == std::shared_ptr < ADTTreeIndex::MatchResults > ())
		{
		   INFO("Error while running search.");
		   return output;
		}

		for(unsigned int i =0; i< matches_index->matches_ids.size(); i++)
		{
		    output.push_back(matches_index->matches_ids[i]);
		}
		
		double end_time_query = CycleTimer::currentSeconds();
		std::stringstream ss;
		ss.str("");
		ss << "Query Processing time is :";
		ss << (end_time_query- start_time_query)*1000;
		ss << " milli seconds";
		INFO(ss.str());

		totalTime += end_time_query- start_time_query;
		// validate matches
		cv::Mat im = cv::imread(query_image->get_location(),cv::IMREAD_GRAYSCALE);
		cv::Mat keypoints_0,descriptors_0;
		ADTVision::compute_sparse_akaze_feature(im,keypoints_0,descriptors_0);

		std::vector<int> validated(MIN(num_validate, matches_index->matches_ids.size()), 0);
		total_tested += validated.size();

		uint32_t	total_correct_tmp = 0;

		for(uint32_t j = 0; j < validated.size(); j++)
		{
			cv::Mat keypoints_1, descriptors_1;
			std::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(userdataset->image(matches_index->matches_ids[j]));
			
			const std::string & match_keypoints_location = userdataset->feat_location(match_image->feature_path("keypoints"));
			const std::string & match_descriptors_location = userdataset->feat_location(match_image->feature_path("descriptors"));
			
			adtfilesystem::load_cvmat(match_keypoints_location, keypoints_1);
			adtfilesystem::load_cvmat(match_descriptors_location, descriptors_1);

			cv::detail::MatchesInfo match_info;
			ADTVision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

			validated[j] = ADTVision::is_good_match(match_info) ? 1 : -1;
			
			if(validated[j] > 0) 
			{
				total_correct_tmp++;
			}
		}

#ifdef EVALUATE	
		ADTMatchesPage html_output_index;
		html_output_index.add_match(query_image,matches_index->matches_ids,userdataset,std::make_shared < std::vector<int> > (validated));

		std::stringstream	outfilestr;
		outfilestr << userdataset->location() << "/results/matches/tree_" << ti->tree_splits()<<"_"<<ti->tree_depth();
		html_output_index.write(outfilestr.str());

#endif
	}

	return output;
}

bool  ADTRetrieval::deletUser(std::string emailid)
{	
	std::stringstream ss;
	ss.str("");
	ss << "Deleting the User with email id :" << emailid;
	INFO(ss.str());

	return userMgt->deleteUser(emailid);
}

bool  ADTRetrieval::deletUser(int userid)
{	
	std::stringstream ss;
	ss.str("");
	ss << "Deleting the User with user id :" << userid;
	INFO(ss.str());
	return userMgt->deleteUser(userid);
}

bool ADTRetrieval::testRandomImages(const std::shared_ptr<ADTDataset> &datasetTest,std::vector<std::shared_ptr<const ADTImage> > &random_images)
{
	uint32_t num_validate = 10;
	uint32_t numberofImages_test = random_images.size();
   if(retrievalMethod == INVERTEDINDEX)
   {
		// search index
		INFO("Running index search");
		
		ADTMatchesPage html_output_index;
		uint32_t	total_correct = 0, total_tested = 0;
		double Score = 0;
		for(uint32_t i = 0; i < numberofImages_test; i++)
		{
			std::shared_ptr<const SimpleDataset::SimpleImage> query_image = std::static_pointer_cast <const SimpleDataset::SimpleImage > (random_images[i]);
			INFO("Searching for " + query_image->get_location());
			std::shared_ptr<ADTInvertedIndex::MatchResults>	matches_index = std::static_pointer_cast <ADTInvertedIndex::MatchResults >	(ii->search(std::shared_ptr < ADTInvertedIndex::SearchParams > (),datasetTest,query_image));

			if(matches_index == std::shared_ptr < ADTInvertedIndex::MatchResults > ())
			{
				INFO("Error while running search.");
				continue;
			}
			Score += getScore(query_image,matches_index);
			// validate matches
			cv::Mat keypoints_0, descriptors_0;
			const std::string & query_keypoints_location = datasetTest->feat_location(query_image->feature_path("keypoints"));
			const std::string & query_descriptors_location = datasetTest->feat_location(query_image->feature_path("descriptors"));
			adtfilesystem::load_cvmat(query_keypoints_location, keypoints_0);
			adtfilesystem::load_cvmat(query_descriptors_location, descriptors_0);
			std::vector<int> validated(MIN(num_validate, matches_index->matches_ids.size()), 0);
			total_tested += validated.size();

			uint32_t	total_correct_tmp = 0;

			for(uint32_t j = 0; j < validated.size(); j++)
			{
				cv::Mat keypoints_1, descriptors_1;
				std::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(datasetTest->image(matches_index->matches_ids[j]));
			
				const std::string & match_keypoints_location = datasetTest->feat_location(match_image->feature_path("keypoints"));
				const std::string & match_descriptors_location = datasetTest->feat_location(match_image->feature_path("descriptors"));
			
				adtfilesystem::load_cvmat(match_keypoints_location, keypoints_1);
				adtfilesystem::load_cvmat(match_descriptors_location, descriptors_1);

				cv::detail::MatchesInfo match_info;
				ADTVision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

				validated[j] = ADTVision::is_good_match(match_info) ? 1 : -1;
			
				if(validated[j] > 0) 
					total_correct_tmp++;
			}

			total_correct += total_correct_tmp;
			html_output_index.add_match(query_image,matches_index->matches_ids,datasetTest,std::make_shared < std::vector<int> > (validated));

			std::stringstream	outfilestr;
			outfilestr << datasetTest->location() << "/results/matches/index_" << bow->num_clusters();
			html_output_index.write(outfilestr.str());
		}
	
		std::stringstream ss;
		ss.str("");
		ss << "UKbench Score is: ";
		ss << Score/numberofImages_test;
		INFO(ss.str());
   }
   else
   {
		// search index
		INFO("Running Tree search");
		
		ADTMatchesPage html_output_index;
		uint32_t	total_correct = 0, total_tested = 0;

		std::shared_ptr<ADTTreeIndex::SearchParams> vt_params =  std::make_shared<ADTTreeIndex::SearchParams>();
		vt_params->amountToReturn = 64;
		double Score = 0;
		double totalTime =0;
		for(uint32_t i = 0; i < numberofImages_test; i++)
		{
			double start_time_query = CycleTimer::currentSeconds();
			std::shared_ptr<const SimpleDataset::SimpleImage> query_image = std::static_pointer_cast <const SimpleDataset::SimpleImage > (random_images[i]);
			INFO("Searching for " +query_image->get_location());
			std::shared_ptr<ADTTreeIndex::MatchResults>	matches_index = std::static_pointer_cast <ADTTreeIndex::MatchResults >	(ti->search2(vt_params,datasetTest, query_image));

			if(matches_index == std::shared_ptr < ADTTreeIndex::MatchResults > ())
			{
				INFO("Error while running search.");
				continue;
			}
			
			double end_time_query = CycleTimer::currentSeconds();
			
			std::stringstream ss;
			ss.str("");
			ss << "Query Processing time is :";
			ss << (end_time_query- start_time_query)*1000;
			ss << " milli seconds";
			INFO(ss.str());

			totalTime += end_time_query- start_time_query;

			Score += getScore(query_image,matches_index);
			// validate matches
			cv::Mat keypoints_0, descriptors_0;
			const std::string & query_keypoints_location = datasetTest->feat_location(query_image->feature_path("keypoints"));
			const std::string & query_descriptors_location = datasetTest->feat_location(query_image->feature_path("descriptors"));
			adtfilesystem::load_cvmat(query_keypoints_location, keypoints_0);
			adtfilesystem::load_cvmat(query_descriptors_location, descriptors_0);
			std::vector<int> validated(MIN(num_validate, matches_index->matches_ids.size()), 0);
			total_tested += validated.size();

			uint32_t	total_correct_tmp = 0;

			for(uint32_t j = 0; j < validated.size(); j++)
			{
				cv::Mat keypoints_1, descriptors_1;
				std::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(datasetTest->image(matches_index->matches_ids[j]));
			
				const std::string & match_keypoints_location = datasetTest->feat_location(match_image->feature_path("keypoints"));
				const std::string & match_descriptors_location = datasetTest->feat_location(match_image->feature_path("descriptors"));
			
				adtfilesystem::load_cvmat(match_keypoints_location, keypoints_1);
				adtfilesystem::load_cvmat(match_descriptors_location, descriptors_1);

				cv::detail::MatchesInfo match_info;
				ADTVision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

				validated[j] = ADTVision::is_good_match(match_info) ? 1 : -1;
			
				if(validated[j] > 0) 
					total_correct_tmp++;
			}

			total_correct += total_correct_tmp;
			html_output_index.add_match(query_image,matches_index->matches_ids,datasetTest,std::make_shared < std::vector<int> > (validated));

			std::stringstream	outfilestr;
			outfilestr << datasetTest->location() << "/results/matches/tree_" << ti->tree_splits()<<"_"<<ti->tree_depth();
			html_output_index.write(outfilestr.str());
		}
		std::stringstream ss;
		ss.str("");
		ss << "UKbench Score is: ";
		ss << Score/numberofImages_test;
		INFO(ss.str());
		ss.str("");
		ss << "Average query Processing time is: ";
		ss << 1000*totalTime/numberofImages_test;
		ss << " milli seconds";
		INFO(ss.str());
   }

   return true;
}

bool ADTRetrieval::addtoDatabase(const std::shared_ptr<ADTDataset> &datasetTest)
{
	if(INVERTEDINDEX == retrievalMethod)
	{
		INFO("Adding images to the database: Inverted Index");
		std::shared_ptr<ADTInvertedIndex::TrainParams> train_params = std::make_shared < ADTInvertedIndex::TrainParams > ();
		train_params->bag_of_words = bow;
		ii->addtoDatabase(datasetTest, train_params, datasetTest->all_images());
	}
	else
	{
		std::stringstream indexed_database_location("");
		indexed_database_location << datasetTest->location() << "/indexed/" << "index_"<< ".invertedIndex";

		if(!adtfilesystem::file_exists(indexed_database_location.str()))
		{
			INFO("Adding images to the database: Vocabulary Index");
			ti->addtoDatabase(datasetTest,datasetTest->all_images());
			ti->save_db(indexed_database_location.str());
		}
		else
		{
			ti->load_db(indexed_database_location.str());
		}
	}
	return true;
}

void ADTRetrieval::train_bow(const std::shared_ptr<ADTDataset> &dataset)
{
	std::stringstream vocab_output_file;
	vocab_output_file << dataset->location() << "/vocabulary/" << "index_"<<bow->num_clusters() << ".invertedIndex";

	if(adtfilesystem::file_exists(vocab_output_file.str()))
	{
		bow->load(vocab_output_file.str());
	}
	else
	{
		bow->train(dataset);
		bow->save(vocab_output_file.str());
	}
}
