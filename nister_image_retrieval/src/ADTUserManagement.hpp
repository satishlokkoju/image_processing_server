#pragma once

/*satish.lokkoju@gmail.com*/
/*11/1/2016*/

/*user management*/

#include <stdio.h>
#include <iostream>
#include <fstream>

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

#include <boost/bimap.hpp>
#include <boost/functional/hash.hpp>

typedef boost::bimap<std::string,uint64> user_list_type;

class ADTUserManagement
{
public:
		ADTUserManagement(std::string users_folder)
		{
			usersFolder = users_folder;
			std::stringstream ss;
			ss.str("");
			ss << users_folder;
			ss << "/users.yml.gz";
			usersFolder_filename = ss.str();
			readUserList();
		};

		bool addUser(std::string emailid);
		void readUserList(std::string filePath);
		void writeUserList(std::string filepath);
		void readUserList();
		void writeUserList();
		bool findUser(std::string emailid);
		bool findUser(uint64 userid);
		bool deleteUser(std::string emailid);
		bool deleteUser(uint64 userid);
		bool addUser(uint64 user_id);
		std::string getUserLocation(uint64 user_id);
		std::string getUserLocation(std::string emailid);
private:
		boost::bimap<std::string,uint64> userList;
		boost::hash<std::string> string_hash;
		std::string usersFolder;
		std::string usersFolder_filename;
};
