

#include "ADTUserManagement.hpp"

bool ADTUserManagement::addUser(std::string emailid)
{
	std::stringstream ss;
	
	if(findUser(emailid))
	{
		ss.str("");
		ss << "User with emailid " << emailid << " Already exists  !";
		INFO(ss.str());
		return false;
	}

	size_t userid = string_hash(emailid);
	ss.str("");
	ss << usersFolder;
	ss << "/user_";
	ss << emailid;
	ss << "/images/";
	adtfilesystem::create_file_directory(ss.str());
		
	ss.str("Creating user at directory: ");
	INFO(ss.str());

	userList.insert(boost::bimap<std::string, uint64>::value_type(emailid,userid));
	writeUserList();
	return true;
}

bool ADTUserManagement::addUser(uint64 user_id)
{
	std::stringstream ss;
	if(findUser(user_id))
	{
		ss.str("");
		ss << "User with userid " << user_id << " Already exists  !";
		INFO(ss.str());
		return false;
	}
	ss.str("");
	ss << user_id;
	ss << "@google.com";
	std::string emailid = ss.str();

	ss.str("");
	ss << usersFolder;
	ss << "/user_";
	ss << user_id;
	ss << "/images/";
	adtfilesystem::create_file_directory(ss.str());

	ss.str("");
	ss << "Adding Image to user_id :";
	ss << user_id <<std::endl;
	ss << "User location :";
	ss << usersFolder << std::endl;
	INFO(ss.str());

	userList.insert(boost::bimap<std::string, uint64>::value_type(emailid,user_id));
	writeUserList();
	return true;
}

std::string ADTUserManagement::getUserLocation(uint64 user_id)
{
	std::string path = "";
	std::stringstream ss;
	if(!findUser(user_id))
	{
		ss.str("");
		ss << "User with user id " << user_id << " doesnt exist  !";
		INFO(ss.str());
		return path;
	}
	ss.str("");
	ss << usersFolder;
	ss << "/user_";
	ss << user_id;
	return ss.str();
}

std::string ADTUserManagement::getUserLocation(std::string emailid)
{
	std::string path = "";
	std::stringstream ss;
	if(!findUser(emailid))
	{
		ss.str("");
		ss << "User with email id " << emailid << " doesnt exist  !";
		INFO(ss.str());
		return path;
	}

	user_list_type::left_const_iterator iter = userList.left.find(emailid);
	
	uint64 user_id = iter->second;
	ss.str("");
	ss << usersFolder;
	ss << "/user_";
	ss << emailid;
	return ss.str();
}

void ADTUserManagement::readUserList(std::string filePath)
{
	userList.clear();
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
	
	cv::FileNode fdb = fs["users_db"];
	cv::FileNode fn = fdb["users"];
	for(uint32_t wid = 0; wid < fn.size(); ++wid)
	{
		cv::FileNode fw = fn[wid];
    
		for(unsigned int i = 0; i < fw.size(); ++i)
		{
			std::string  email_id = (std::string)fw[i]["email_id"];
			int user_id= (int)fw[i]["user_id"];
			userList.insert(boost::bimap<std::string, uint64>::value_type(email_id,user_id));
		}
	}	

	fs.release();
}

void ADTUserManagement::writeUserList(std::string filepath)
{
	//create the user list file.
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
	std::stringstream ss;
	ss.str("");
	fs << "users_db" <<"{";
	fs << "usermanagemenet" << "serverside";
	fs << "users" <<"[";
	
	for( user_list_type::const_iterator iter = userList.begin(), iend = userList.end(); iter != iend; ++iter )
	{
		ss.str("");
		ss <<iter->right;
		fs << "{:"<<"email_id" << iter->left<< "user_id"<<ss.str()<<"}";
		std::cout << iter->left ;
		std::cout << " ";
		std::cout << iter->right << std::endl;
	}
	
	fs << "]";
	fs << "}"; //user db;

	fs.release();
}

void ADTUserManagement::readUserList()
{
	userList.clear();
    cv::FileStorage fs(usersFolder_filename, cv::FileStorage::READ);
	
	if(!fs.isOpened())
		return;

	cv::FileNode fdb = fs["users_db"];
	cv::FileNode fw = fdb["users"];
	for(uint32_t wid = 0; wid < fw.size(); ++wid)
	{
			std::string  email_id = (std::string)fw[wid]["email_id"];
			std::string user_id_str= (std::string)fw[wid]["user_id"];
			uint64 user_id = std::stoull(user_id_str);
			userList.insert(boost::bimap<std::string, uint64>::value_type(email_id,user_id));
			std::cout << email_id;
			std::cout << " ";
			std::cout << user_id << std::endl;
	}	

	fs.release();
}

void ADTUserManagement::writeUserList()
{
	//create the user list file.
    cv::FileStorage fs(usersFolder_filename, cv::FileStorage::WRITE);
	std::stringstream ss;
	ss.str("");
	fs << "users_db" <<"{";
	fs << "usermanagemenet" << "serverside";
	fs << "users" <<"[";
	
	for( user_list_type::const_iterator iter = userList.begin(), iend = userList.end(); iter != iend; ++iter )
	{
		ss.str("");
		ss <<iter->right;
		fs << "{:"<<"email_id" << iter->left<< "user_id"<<ss.str()<<"}";
	}
	
	fs << "]";
	fs << "}"; //user db;

	fs.release();
}
bool ADTUserManagement::findUser(std::string emailid)
{
	if(userList.empty())
		return false;

	if (userList.left.find(emailid) == userList.left.end()) 
		return false;

	return true;
}

bool ADTUserManagement::findUser(uint64 userid)
{
	if(userList.empty())
		return false;

	if (userList.right.find(userid) == userList.right.end()) 
		return false;

	return true;
}

bool ADTUserManagement::deleteUser(std::string emailid)
{
	if(userList.empty())
		return false;

	user_list_type::left_const_iterator  iter = userList.left.find(emailid);
	if ( iter == userList.left.end()) 
		return false;
	
	userList.left.erase(emailid);
	writeUserList();
	return true;
}

bool ADTUserManagement::deleteUser(uint64 userid)
{
	if(userList.empty())
		return false;

	user_list_type::right_const_iterator  iter = userList.right.find(userid);
	if ( iter == userList.right.end()) 
		return false;
	
	userList.right.erase(userid);
	writeUserList();
	return true;
}
