/*satish.lokkoju@gmail.com*/
/*25/07/2015*/

/*Wrapper function to test the CBIR*/
/*23/08/2015*/
/*additional functionality has been added*/
/*Support for Vocabulary Tree based search is the major feature addition*/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "ADTRetrieval.hpp"

#include <boost/current_function.hpp> 
void wait()
{
  INFO("Press enter to continue");
  getchar();
}

void usage(char **argv)
{
	INFO("      USAGE:");
	INFO("            ");
	INFO(argv[0]);
	INFO("    <path to dir containing images folder>");
	INFO("or")
	INFO(argv[0]);
	INFO("    <path to dir containing folder containing user image management>");
	INFO("	The users folder has the following format");
	INFO("	/user_1/images/");
	INFO("	/user_2/images/");
	INFO("	....");
}

/*usage*/
/*The below wrapper can be used to test INverted index based file search and*/
/*Vocabulary tree based file search. Please change the 2nd parameter in ADTRetrieval accordingly*/
/*Accepted values InvertedIndex or VocabularyTree */
int main_2(int argc,char **argv)
{
	if(argc !=2)
	{
		usage(argv);
		exit(1);
	}

	INFO("/*****************************************************/");
	INFO("/*************Image Retrieval Engine******************/");
	INFO("/*****************************************************/");
	// parameters
	uint32_t num_train_images = 1000;
	uint32_t num_test_images  = 128;
	
	std::string descType;
	std::shared_ptr<SimpleDataset> datasetTrain;
	//std::shared_ptr<SimpleDataset> datasetTest;
	
	descType = "SURF";
	//datasetTrain = std::make_shared<SimpleDataset>(UKBENCHLOCATION,UKBENCHLOCATION_DATABASE,descType);
	datasetTrain = std::make_shared<SimpleDataset>(std::string(argv[1]),std::string(argv[1])+std::string("/db_release.db"),descType);
	ADTRetrieval retrival(num_train_images,VOCABULARYTREE,descType);
	retrival.learn(datasetTrain);
	
	//datasetTest = std::make_shared<SimpleDataset>(UKBENCHLOCATION_TEST,UKBENCHLOCATION_DATABASE_TEST,descType);
	retrival.addtoDatabase(datasetTrain);
	
  uint32_t	numberofImages_test = MIN(datasetTrain->num_images(), num_test_images);
	std::vector<std::shared_ptr<const ADTImage>> random_images = datasetTrain->random_images(numberofImages_test);
	retrival.testRandomImages(datasetTrain,random_images);
	wait();
	return 0;
}


using namespace cv;

/*usage*/
/*The below wrapper can be used to test INverted index based file search and*/
/*Vocabulary tree based file search. Please change the 4th parameter in ADTRetrieval accordingly*/
/*Accepted values InvertedIndex or VocabularyTree */
int main(int argc,char **argv)
{
#if 0
	// parameters
	uint32_t num_train_images = 2000;
	uint32_t num_test_images  = 500;

	
	std::string descType;
	std::shared_ptr<SimpleDataset> datasetTrain;
	std::shared_ptr<SimpleDataset> datasetTest;
	
	/*
	double start = CycleTimer::currentSeconds();
	Sleep(10000);
	double end = CycleTimer::currentSeconds();
	std::cout  << (end-start);
	std::cout << " seconds" << std::endl;
	*/

	descType = "AKAZE";
	//datasetTrain = std::make_shared<SimpleDataset>(UKBENCHLOCATION_MINI,UKBENCHLOCATION_DATABASE_MINI,descType);
	datasetTrain = std::make_shared<SimpleDataset>(UKBENCHLOCATION,UKBENCHLOCATION_DATABASE,descType);
	ADTRetrieval retrival(num_train_images,VOCABULARYTREE,descType);
	retrival.learn(datasetTrain);
	
	datasetTest = std::make_shared<SimpleDataset>(UKBENCHLOCATION_TEST,UKBENCHLOCATION_DATABASE_TEST,descType);
	retrival.addtoDatabase(datasetTest);
	
    uint32_t	numberofImages_test = MIN(datasetTest->num_images(), num_test_images);
	std::vector<std::shared_ptr<const ADTImage> > random_images = datasetTest->random_images(numberofImages_test);
	retrival.testRandomImages(datasetTest,random_images);
	wait();
	return 0;
#else
#ifdef DUMPGROUNDTRUTH
	//create the Ground truth YAML file
    cv::FileStorage fs("ground_truth.yml.gz", cv::FileStorage::WRITE);
	std::stringstream ss;
	ss.str("");
	fs << "dataset" << "UKBENCH";
	fs << "test_images" <<"[";
	for(int i =0; i< 10200; i++)
	{
		ss.str("");
		ss <<std::setw(5) << std::setfill('0') << i;
		int a = (i/4)*4;
		int b = a+1;
		int c = b+1;
		int d = c+1;
		std::vector<int> res;
		if(a!=i)
		{
			res.push_back( a);
		}
		if(b!=i)
		{
			res.push_back( b);
		}
		
		if(c!=i)
		{
			res.push_back( c);
		}

		if(d!=i)
		{
			res.push_back( d);
		}
		fs << "{:"<<"query" << ss.str()<< "results"<< "[:";
		ss.str("");
		ss << std::setw(5) << std::setfill('0') << res[0];
		fs << ss.str();
		ss.str("");
		ss << std::setw(5) << std::setfill('0') << res[1];
		fs << ss.str();			
		ss.str("");
		ss << std::setw(5) << std::setfill('0') << res[2];
		fs << ss.str();
		fs << "]"<<"}";
	}
	
	fs << "]";
	fs.release();

    cv::FileStorage fs_in("ground_truth.yml.gz", cv::FileStorage::READ);
	
    cv::FileNode n = fs_in["test_images"];
    if (n.type() != cv::FileNode::SEQ)
    {
		std::cerr << "images is not a sequence! FAIL" << std::endl;
		return 1;
    }

    std::cout << "reading images\n";
    cv::FileNodeIterator it = n.begin(), it_end = n.end();
	  int i=0;
    for (; it != it_end; ++it)
    {
		std::cout << "\n";
		std::cout << "Query: " << std::endl;
		std::cout << (std::string)(*it)["query"] <<std::endl;
		cv::FileNode results = (*it)["results"];
		
		std::cout << "Results: " << std::endl;
		cv::FileNodeIterator it_results = results.begin(), it_results_end = results.end();
		for (; it_results != it_results_end; ++it_results)
		{
			std::cout << (std::string)(*it_results) << std::endl;
		}

		i++;
    }

	fs_in.release();

#endif

	// parameters
	uint32_t num_train_images = 1000;
	uint32_t num_test_images  = 200;

	
	std::string descType;
	std::shared_ptr<SimpleDataset> datasetTrain;
	std::shared_ptr<SimpleDataset> datasetTest;
	
	/*
	double start = CycleTimer::currentSeconds();
	Sleep(10000);
	double end = CycleTimer::currentSeconds();
	std::cout  << (end-start);
	std::cout << " seconds" << std::endl;
	*/

	descType = "AKAZE";
	datasetTrain = std::make_shared<SimpleDataset>(UKBENCHLOCATION,UKBENCHLOCATION_DATABASE,descType);
	ADTRetrieval retrival(num_train_images,VOCABULARYTREE,descType,USERS_LOCATION);
	retrival.learn(datasetTrain);

	// add the user databases
	//example initialize a user given its database location and userid
	retrival.addUser(0000,USER0_LOCATION);
	retrival.addUser(0001,USER1_LOCATION);
	retrival.addUser(0003,USER3_LOCATION);
	retrival.addUser(0002,USER2_LOCATION);

	// intialize an user using his/her email id
	retrival.addUser("satish.l@google.com");
	retrival.addUser("amogh@google.com");
	retrival.addUser("rkant@google.com");

	for(int i =0; i<500; i++)
	{
		std::stringstream ss("");
		ss << "/home/satishl/satish/datasets/ukbench/images/ukbench";
		ss << std::setw(5) << std::setfill('0') << i << ".jpg";
		retrival.addImageUser("satish.l@google.com",ss.str());
	}
	
	for(int i =200; i<300; i++)
	{
		std::stringstream ss("");
		ss << "/home/satishl/satish/datasets/ukbench/images/ukbench";
		ss << std::setw(5) << std::setfill('0') << i << ".jpg";
		retrival.addImageUser("rkant@google.com",ss.str());
	}

		
#ifdef EVALUATE
	// perform search on the users -Regression testing

	retrival.searchUserDB_testImages(0,USERS_LOCATION);
	retrival.searchUserDB_testImages("satish.l@google.com",USERS_LOCATION);

#endif
	retrival.addImageUser(0001,"/home/satishl/satish/datasets/ukbench/images/ukbench00057.jpg");
	retrival.addImageUser(0002,"/home/satishl/satish/datasets/ukbench/images/ukbench00087.jpg");
	retrival.addImageUser(0003,"/home/satishl/satish/datasets/ukbench/images/ukbench00027.jpg");

	cv::Mat queryImage = cv::imread("/home/satishl/satish/datasets/ukbench/images/ukbench00036.jpg");

	retrival.retrieveImage(000,queryImage,3);
	retrival.retrieveImage(002,queryImage,4);
	retrival.retrieveImage(003,queryImage,5);

	retrival.retrieveImage("satish.l@google.com",queryImage,3);
	retrival.retrieveImage("satish.l@google.com",queryImage,4);
	//retrival.retrieveImage("rkant@google.com",queryImage,5);

	// remove image from a particular user using image id
	retrival.removeImageUser(0001,001);
	retrival.removeImageUser(0002,002);
	retrival.removeImageUser(0003,003);
	
	// remove image from a particular user using image id
	retrival.removeImageUser("satish.l@google.com",001);
	retrival.removeImageUser("satish.l@google.com",002);
	retrival.removeImageUser("satish.l@google.com",003);

	// delete user using email id
	retrival.deletUser("satish.l@google.com");
	retrival.deletUser("amogh@google.com");	
	retrival.deletUser("rkant@google.com");

	// delete User using Userid
	retrival.deletUser(0003);
	retrival.deletUser(0004);	
	retrival.deletUser(0005);
	
	wait();
#endif

}



