#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*file system wrapper containing utilties for directory read/write,file read write*/
/*etc.*/

#include "ADTfilesystem.hpp"
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>


namespace adtfilesystem 
{
	bool copy_file(const std::string& input_path,const std::string &output_path)
	{
		const boost::filesystem::path source(input_path);
		const boost::filesystem::path dest(output_path);
		create_file_directory(output_path);
		bool status = true;
		try
                {
			boost::filesystem::copy_file(source,dest,boost::filesystem::copy_option::fail_if_exists);
		}
		catch (const boost::filesystem::filesystem_error& e)
		{
			status = false;
		}
		return status;
	}

	bool delete_file(const std::string& input_path)
	{
		const boost::filesystem::path source(input_path);
		bool status = true;
		try
                {
        		if(boost::filesystem::exists(source)) 
				boost::filesystem::remove(source);
		}
		catch (const boost::filesystem::filesystem_error& e)
		{
			status = false;
		}
		return status;
	}

	uint64_t secondsSinceEpoch(const boost::posix_time::ptime& time) 
	{
		const boost::posix_time::ptime epoch((boost::gregorian::date(1970,1,1)));
		boost::posix_time::time_duration duration = time - epoch;
		return duration.total_seconds();
	}

	std::vector<std::string> list_updated_files(const std::string &path,uint64_t oldTimeinSecsSinceEpoch, const std::string &ext)
	{
		std::vector<std::string> new_files;
		std::stringstream ss("");
		ss << path;
		ss << "\\images\\";
		boost::filesystem::path dir(ss.str());
		boost::posix_time::ptime nowTime(boost::posix_time::second_clock::local_time());
		boost::posix_time::time_duration  timedurationinseconds(0,0,oldTimeinSecsSinceEpoch);
		boost::posix_time::ptime oldTime(boost::gregorian::date(1970,1,1),timedurationinseconds);
		std::cout << oldTime << std::endl;
		boost::filesystem::directory_iterator dirIter( dir ), dirIterEnd;
		while ( dirIter != dirIterEnd )
		{
			if ( boost::filesystem::exists( *dirIter ) && !boost::filesystem::is_directory( *dirIter ) )
			{
				std::time_t t = boost::filesystem::last_write_time( *dirIter );
				boost::posix_time::ptime lastAccessTime = boost::posix_time::from_time_t( t );
				
				//std::cout << (*dirIter).path() <<std::endl;
				//std::cout << lastAccessTime << std::endl;
				//std::cout << t << std::endl;
				if ( lastAccessTime >= oldTime && lastAccessTime <= nowTime )
				{
					std::cout << (*dirIter).path() <<std::endl;
					new_files.push_back((*dirIter).path().string());
				}     
			}
			++dirIter;
		}
 		return new_files;
	}

	bool is_directory(const std::string& input_path)
	{
		return boost::filesystem::is_directory(input_path);
	}

	bool file_exists(const std::string& name) 
	{
	  struct stat buffer;
	  return (stat (name.c_str(), &buffer) == 0);
	}
	
	void create_file_directory(const std::string &absfilepath) 
	{
		boost::filesystem::path p(absfilepath.c_str());
		boost::filesystem::path d = p.parent_path();
		if(!boost::filesystem::exists(d)) 
		{
			boost::filesystem::create_directories(d);
		}
	}
	
	struct cvmat_header 
	{
		uint64_t elem_size;
		int32_t elem_type;
		uint32_t rows, cols;
	};

	bool write_cvmat(const std::string &fname, const cv::Mat &data) 
	{
		std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
		cvmat_header h;
		h.elem_size = data.elemSize();
		h.elem_type = data.type();
		h.rows = data.rows;
		h.cols = data.cols;
		ofs.write((char *)&h, sizeof(cvmat_header));
		ofs.write((char *)data.ptr(), h.rows * h.cols * h.elem_size);
		return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	bool load_cvmat(const std::string &fname, cv::Mat &data) 
	{
		if(!file_exists(fname)) 
			return false;
	
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		cvmat_header h;
		ifs.read((char *)&h, sizeof(cvmat_header));
		
		if (h.rows == 0 || h.cols == 0) 
			return false;

		data.create(h.rows, h.cols, h.elem_type);
		
		ifs.read((char *)data.ptr(), h.rows * h.cols * h.elem_size);
		return (ifs.rdstate() & std::ifstream::failbit) == 0;
	}

	bool write_sparse_vector(const std::string &fname, const std::vector<std::pair<uint32_t, float > > &data) 
	{
		std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
		uint32_t dim0 = data.size();
		ofs.write((char *)&dim0, sizeof(uint32_t));
		ofs.write((char *)&data[0], sizeof(std::pair<uint32_t, float >) * dim0);
		return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	bool load_sparse_vector(const std::string &fname, std::vector<std::pair<uint32_t, float > > &data) 
	{
		if(!file_exists(fname))
		{
			// INFO("couldnt find " << fname;
		 return false;
		}
	
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		uint32_t dim0;
		ifs.read((char *)&dim0, sizeof(uint32_t));
		data.resize(dim0);
		ifs.read((char *)&data[0], sizeof(std::pair<uint32_t, float >) * dim0);
		return (ifs.rdstate() & std::ifstream::failbit) == 0;
	}

	std::vector<std::string> list_files(const std::string &path, const std::string &ext, bool recursive) 
	{
		boost::filesystem::path input_path(path);
		std::vector<std::string> file_list;
		if (boost::filesystem::exists(input_path) && boost::filesystem::is_directory(input_path)) 
		{
			if(recursive) 
			{
				boost::filesystem::recursive_directory_iterator end;
				for (boost::filesystem::recursive_directory_iterator it(input_path); it != end; ++it) 
				{
					if (!boost::filesystem::is_directory(*it)) 
					{
						if ((ext.length() > 0 && boost::filesystem::extension(*it) == boost::filesystem::path(ext)) || ext.length() == 0) 
						{
							file_list.push_back(it->path().string());
						}
					}
				}
			}
			else 
			{
				boost::filesystem::directory_iterator end;
				for (boost::filesystem::directory_iterator it(input_path); it != end; ++it) 
				{
					if (!boost::filesystem::is_directory(*it)) 
					{
						if ((ext.length() > 0 && boost::filesystem::extension(*it) == boost::filesystem::path(ext)) || ext.length() == 0) 
						{
								file_list.push_back(it->path().string());
						}
					}
				}
			}
		}

		return file_list;
	}
	
	std::vector<std::string> add_users(const std::string &path, const std::string &user) 
	{
		boost::filesystem::path input_path(path);
		std::vector<std::string> user_list;

		if (boost::filesystem::exists(input_path) && boost::filesystem::is_directory(input_path)) 
		{
			boost::filesystem::recursive_directory_iterator end;
			for (boost::filesystem::recursive_directory_iterator it(input_path); it != end; ++it) 
			{
				if (boost::filesystem::is_directory(*it)) 
				{
					std::string userpath = it->path().string();
					userpath.find_last_of('\\');

					user_list.push_back(userpath);	
				}
			}
		}
		return user_list;
	}

	std::string basename(const std::string &path, bool include_extension) 
	{
		if (!include_extension)
			return boost::filesystem::basename(boost::filesystem::path(path));

		return boost::filesystem::basename(boost::filesystem::path(path)) + boost::filesystem::extension(path);
	}

	bool write_text(const std::string &fname, const std::string &text) 
	{
		std::ofstream ofs(fname, std::ios::trunc);
		ofs.write(text.c_str(), text.size());
	
		return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	bool write_vector(const std::string &fname, const std::vector<float> &data) 
	{
	    // create_file_directory(fname);

	    std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
	   	uint32_t dim0 = data.size();
		ofs.write((char *)&dim0, sizeof(uint32_t));
	    ofs.write((char *)&data[0], data.size()*sizeof(float));
	    return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	uint64_t last_writetime(const std::string &foldername)
	{
		time_t time= std::time( 0 ) ;
		boost::filesystem::last_write_time(foldername,time);
		return time;
	}

	bool load_vector(const std::string &fname, std::vector<float> &data) 
	{
		if (!file_exists(fname)) 
			return false;
		
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		uint32_t dim0;
		ifs.read((char *)&dim0, sizeof(uint32_t));
		data.resize(dim0);
		ifs.read((char *)&data[0], data.size()*sizeof(float));

		return (ifs.rdstate() & std::ifstream::failbit) == 0;
	}
}
