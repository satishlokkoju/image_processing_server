#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Simple class to hold all the data images together*/
/*etc.*/

#include <stdint.h>
#include <iostream>
#include <vector>
#include "ADTimages.hpp"
#include "ADTDefines.hpp"
#include "ADTMath.hpp"

#include <memory>
#include <boost/bimap.hpp>
#include <boost/date_time.hpp>
#include <sstream>
#include <iomanip>

typedef boost::bimap<std::string, uint64_t> id_image_bimap_type;

class ADTDataset 
{

public:
	/// Constructs a dataset given a base location.  An example base location might be
	/// /c/data/.  Given this base location, an implementation of the Dataset should 
	/// find all the data and construct a mapping between the data and the id, for example 
	/// by searching through base_location + /images/.
	ADTDataset(const std::string &base_location);

	/// Loads a dataset from the db_data_location.  The base_location provides the absolute
	/// path of data.
	ADTDataset(const std::string &base_location, const std::string &db_data_location);

	ADTDataset();

	virtual ~ADTDataset();

	/// Writes the dataset mapping to the input data location.  Returns true if successful, false
	/// otherwise.
	virtual bool write(const std::string &db_data_location) = 0;

	/// Reads the dataset mapping from the input data location.  Returns true if successful, false
	/// otherwise.
	virtual bool read (const std::string &db_data_location) = 0;

	/// Given a unique integer ID, returns an Image associated with that ID.
	virtual std::shared_ptr<ADTImage> image(uint64_t id) const	= 0;

	/// Returns the number of images in the dataset.
	virtual uint64_t num_images() const = 0;
	
	/// Returns the database id.
	virtual uint64_t database_id() const = 0;

	/// Returns the absolute path of the data directory
	std::string location() const;

	/// Returns the absolute path of the file (appends the file path to the database path).
	std::string image_location(const std::string &relative_path) const;
	// returns the absolute location of the feature descriptor /keypoints file
	std::string feat_location(const std::string &feature_location) const;

	/// Adds the given image to the database, if there is an id collision, will not add the image and 
	/// return false, otherwise returns true.
	virtual bool add_image(const std::shared_ptr<const ADTImage> &image) = 0 ;

	/// Returns a vector of all images in the dataset.
	std::vector<  std::shared_ptr< const ADTImage> > all_images() const;

	/// Returns a vector of random images in the dataset of size count.
	std::vector<  std::shared_ptr< const ADTImage> > random_images(size_t count) const;

	/// @TODO: Shards the dataset to the new input locations, and returns the sharded datasets
	std::vector<ADTDataset> shard(const std::vector<std::string> &new_locations);

	virtual adtmath::sparse_vector_t load_bow_feature(uint64_t id) const = 0;
	virtual std::vector<float> load_vec_feature(uint64_t id) const = 0;

protected:
	/// Map which holds the image path and id
	boost::bimap<std::string, uint64_t> id_image_map;
	std::string	data_directory;  /// Holds the absolute path of the data.
	uint64_t db_id;
};


std::ostream& operator<< (std::ostream &out, const ADTDataset &dataset);
class SimpleDataset : public ADTDataset 
{
public:
	class SimpleImage : public ADTImage 
	{
		public:
			SimpleImage(const std::string path,const uint64_t imageid);
			std::string feature_path(const std::string &feat_name) const;
			std::string get_location() const;
			std::string get_filename() const;
		protected:
			std::string image_path; 
	};
	
	SimpleDataset();
	SimpleDataset(const std::string &base_location);
	SimpleDataset(const std::string &base_location, const std::string &db_data_location,const std::string &feattype);
	~SimpleDataset();
	bool write(const std::string &db_data_location);
	
	/// Reads the specified SimpleDataset.  See write(const std::string &db_data_location) for
	/// more information about the binary format.  Returns true if success, false otherwise. 
	/// (checks the ifstream error bit).
	bool read(const std::string &db_data_location);

	/// Given a unique integer ID, returns an Image associated with that ID.
	std::shared_ptr<ADTImage> image(uint64_t id) const;

	/// Adds the given image to the database, if there is an id collision, will not add the image and 
	/// return false, otherwise returns true.
	bool add_image(const std::shared_ptr<const ADTImage> &image);
	int64_t add_image(const std::string &image_path);
	int64_t delete_image(const std::string imagename);

	/// Returns the number of images in the dataset.
	uint64_t num_images() const;
	
	/// Returns the database id;
	uint64_t database_id() const;
	/// Returns the corresponding feature path given a feature name (ex. "sift").
	adtmath::sparse_vector_t load_bow_feature(uint64_t id) const;
	std::vector<float> load_vec_feature(uint64_t id) const;
	bool computeFeatures();
private:
	
	/// Constructs the dataset an fills in the image id map.
	adtmath::sparse_vector_t load_bow_feature_cache(uint64_t id) const;
	std::vector<float> load_vec_feature_cache(uint64_t id) const;
	void construct_dataset();
	static std::string feat_type;
	uint64_t image_id_cnt;
};
