#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*file system wrapper containing utilties for directory read/write,file read write*/
/*etc.*/

#define BOOST_NO_CXX11_SCOPED_ENUMS
 
#include <stdint.h>
#include "ADTDefines.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

namespace adtfilesystem 
{
	std::string basename(const std::string &path, bool include_extension = false);
	bool file_exists(const std::string& name);
	void create_file_directory(const std::string &absfilepath);
	bool write_cvmat(const std::string &fname, const cv::Mat &data);
	bool load_cvmat(const std::string &fname, cv::Mat &data);
	/// Writes the BoW feature to the specified location.  First dimension of data is cluster index,
	/// second dimension is TF score.
	bool write_sparse_vector(const std::string &fname, const std::vector<std::pair<uint32_t, float > > &data);
	/// Loads the BoW feature from the specified location.  First dimension of data is cluster index,
	/// second dimension is TF score.
	bool load_sparse_vector(const std::string &fname, std::vector<std::pair<uint32_t, float > > &data);
	/// Lists all files in the given directory with an optional extension.  The extension must include
	/// the dot (ie. ext=".txt").  If recursive is true (default), will recursively enter all directories
	std::vector<std::string> list_files(const std::string &path, const std::string &ext = "", bool recursive = true) ;
	/// Writes a text file to the input file location given the input string.  Returns true if success,false otherwise.
	bool write_text(const std::string &fname, const std::string &text);
	/// Writes the vector to the specified location. 
	bool write_vector(const std::string &fname, const std::vector<float> &data);
	/// Loads vector BoW feature from the specified location.  
	bool load_vector(const std::string &fname, std::vector<float> &data);
	uint64_t secondsSinceEpoch(const boost::posix_time::ptime& time);
	bool is_directory(const std::string& input_path);
	bool copy_file(const std::string& input_path,const std::string &output_path);
	bool delete_file(const std::string& input_path);
	std::vector<std::string> list_updated_files(const std::string &apth,uint64_t oldTimeinSecsSinceEpoch, const std::string &ext = "");
	uint64_t last_writetime(const std::string &foldername);
};
