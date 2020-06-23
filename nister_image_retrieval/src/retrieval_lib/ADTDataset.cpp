#include "ADTDataset.hpp"
#include "ADTfilesystem.hpp"
#include "ADTVision.hpp"

#include <fstream>

ADTDataset::ADTDataset(const std::string &base_location) 
{
	data_directory = base_location;
}

ADTDataset::ADTDataset(const std::string &base_location, const std::string &db_data_location) 
{
	data_directory = base_location;
}

ADTDataset::ADTDataset() 
{ 

}

ADTDataset::~ADTDataset() 
{ 

}

std::string ADTDataset::location() const 
{
	return data_directory;
}

std::string ADTDataset::image_location(const std::string &relative_path) const 
{
	return data_directory + "/images/" + relative_path;
}


std::string ADTDataset::feat_location(const std::string &relative_path) const 
{
	return data_directory + "/" + relative_path;
}

std::vector<std::shared_ptr<const ADTImage> > ADTDataset::all_images() const 
{
	std::vector<std::shared_ptr< const ADTImage> > images;
	for( id_image_bimap_type::const_iterator iter = id_image_map.begin(), iend = id_image_map.end(); iter != iend; ++iter )
	{		
		std::shared_ptr<SimpleDataset::SimpleImage> simpleimage = std::make_shared<SimpleDataset::SimpleImage>(iter->left,iter->right);
		images.push_back(simpleimage);
	}
	return images;
}

std::vector<std::shared_ptr<const ADTImage> > ADTDataset::random_images(size_t count) const 
{
	std::vector<std::shared_ptr< const ADTImage> > all = this->all_images();
	std::random_shuffle(all.begin(), all.end());
	std::vector<std::shared_ptr< const ADTImage> > images(all.begin(), all.begin() + count);
	
	std::stringstream ss;
	ss.str("");
	ss << "The number of random images selected is :";
	ss << images.size();
	INFO(ss.str());
	return images;
}

std::ostream& operator<< (std::ostream &out, const std::shared_ptr<ADTDataset> &dataset) 
{
	out << "Dataset location: " << dataset->location() << ", number of images: " << dataset->num_images();
	return out;
}

SimpleDataset::SimpleDataset(const std::string &base_location) : ADTDataset(base_location) 
{
	this->construct_dataset();
}

SimpleDataset::SimpleDataset(const std::string &base_location, const std::string &db_data_location,const std::string &featype) : ADTDataset(base_location, db_data_location) 
{
	INFO("Caching the descriptors...");
	feat_type = featype;
	image_id_cnt = 0;
	if(!adtfilesystem::file_exists(base_location))
	{
		INFO("Please provide a valid images directory");
		exit(1);
	}
	
	if (adtfilesystem::file_exists(db_data_location)) 
	{
		this->read(db_data_location);
	}
	else 
	{
		this->construct_dataset();
		this->write(db_data_location);
	}

	computeFeatures();
}

SimpleDataset::~SimpleDataset() 
{

}

std::shared_ptr<ADTImage> SimpleDataset::image(uint64_t id) const 
{
	const std::string &image_path = id_image_map.right.at(id);
	std::shared_ptr<ADTImage> current_image = std::make_shared<SimpleImage>(image_path, id);
	return current_image;
}

bool SimpleDataset::computeFeatures()
{
	for( id_image_bimap_type::const_iterator iter = id_image_map.begin(),iend = id_image_map.end(); iter != iend; ++iter )
	{
		std::shared_ptr<SimpleDataset::SimpleImage> simpleimage = std::make_shared<SimpleDataset::SimpleImage>(iter->left,iter->right);
		if(simpleimage == std::shared_ptr < SimpleDataset::SimpleImage > ()) 
			continue;

		const std::string & keypoints_location = feat_location(simpleimage->feature_path("keypoints"));
		const std::string & descriptors_location = feat_location(simpleimage->feature_path("descriptors"));

		if(adtfilesystem::file_exists(keypoints_location) && adtfilesystem::file_exists(descriptors_location))
			continue;

		const std::string & image_path_full = simpleimage->get_location();

		if(!adtfilesystem::file_exists(image_path_full)) 
			continue;

		cv::Mat im = cv::imread(image_path_full, cv::IMREAD_GRAYSCALE);

		cv::Mat keypoints, descriptors;
#ifdef NONFREE		
		if(0==feat_type.compare("SIFT"))
		{
			if(!ADTVision::compute_sparse_sift_feature(im,std::shared_ptr < const ADTVision::SIFTParams > (),keypoints,descriptors)) 
				continue;
		}
		else if(0==feat_type.compare("SURF"))
		{
			if(!ADTVision::compute_sparse_surf_feature(im,keypoints,descriptors)) 
				continue;
		}
		else 
#endif
		if(0==feat_type.compare("AKAZE"))
		{
			if(!ADTVision::compute_sparse_akaze_feature(im,keypoints,descriptors)) 
				continue;
		}
		else if(0==feat_type.compare("KAZE"))
		{
			if(!ADTVision::compute_sparse_kaze_feature(im,keypoints,descriptors)) 
				continue;
		}

		adtfilesystem::create_file_directory(keypoints_location);
		adtfilesystem::create_file_directory(descriptors_location);

		adtfilesystem::write_cvmat(keypoints_location, keypoints);
		adtfilesystem::write_cvmat(descriptors_location, descriptors);
	}
	return true;
}

SimpleDataset::SimpleDataset()
{

}

uint64_t SimpleDataset::database_id() const
{
	return db_id;
}

void SimpleDataset::construct_dataset()
{
	INFO(data_directory);
	const std::vector<std::string> &image_file_paths = adtfilesystem::list_files(data_directory + "/images/", ".jpg");
	
	std::stringstream ss;
	ss.str("");
	ss << "The number of images found are:";
	ss << image_file_paths.size();
	INFO(ss.str());
	for (size_t i = 0; i < image_file_paths.size(); i++) 
	{
		//std::string file_name = image_file_paths[i].substr(data_directory.size()+8, image_file_paths[i].size() - data_directory.size()-8);
		id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type( image_file_paths[i], i));
	}
}

bool SimpleDataset::read(const std::string &db_data_location) 
{
	if (!adtfilesystem::file_exists(db_data_location))
	{
		INFO("Unable to Read Database");
		return false;
	}
	std::ifstream ifs(db_data_location, std::ios::binary);
	
	uint64_t num_images,timeinseconds;
	ifs.read((char *)&num_images, sizeof(uint64_t));
	uint64_t maxvalue =0;
	for (uint64_t i = 0; i < num_images; i++) 
	{	
		uint16_t length;
		ifs.read((char *)&image_id_cnt, sizeof(uint64_t));
		ifs.read((char *)&length, sizeof(uint16_t));

		std::string image_location;
		image_location.resize(length);
		
		if(maxvalue<image_id_cnt)
			maxvalue = image_id_cnt;

		ifs.read((char *)&image_location[0], sizeof(char)* length);
		std::shared_ptr<const SimpleImage> simage = std::make_shared<const SimpleImage>(image_location, image_id_cnt);
		this->add_image(simage);
	}
	// increment the image index to enable addition of new images
	image_id_cnt = maxvalue+1;

	return (ifs.rdstate() & std::ifstream::failbit) == 0;
}

bool SimpleDataset::write(const std::string &db_data_location) 
{
	adtfilesystem::create_file_directory(db_data_location);
	
	std::ofstream ofs(db_data_location, std::ios::binary | std::ios::trunc);
	uint64_t num_images = this->num_images();
	ofs.write((const char *)&num_images, sizeof(uint64_t));

	for( id_image_bimap_type::const_iterator iter = id_image_map.begin(), iend = id_image_map.end(); iter != iend; ++iter )
	{
		const std::string &image_location = iter->left;
		uint64_t image_id =  iter->right;
		uint16_t length = image_location.size();
		ofs.write((const char *)&image_id, sizeof(uint64_t));
		ofs.write((const char *)&length, sizeof(uint16_t));
		ofs.write((const char *)&image_location[0], sizeof(char)* length);
	}

	return (ofs.rdstate() & std::ofstream::failbit) == 0;
}

uint64_t SimpleDataset::num_images() const 
{
	return id_image_map.size();
}

SimpleDataset::SimpleImage::SimpleImage(const std::string path,const uint64_t imageid) : ADTImage(imageid) 
{
	image_path = path;
}

std::string SimpleDataset::feat_type = "";

std::string SimpleDataset::SimpleImage::feature_path(const std::string &feat_name) const 
{
	uint32_t level0 = id >> 20;
	uint32_t level1 = (id - (level0 << 20)) >> 10;

	std::stringstream ss;
	ss << "/feats/" <<feat_type <<"/"<< feat_name << "/" << 
		std::setw(4) << std::setfill('0') << level0 << "/" <<
		std::setw(4) << std::setfill('0') << level1 << "/" <<
		std::setw(9) << std::setfill('0') << id << "." << feat_name;
		
	return ss.str();
}

std::string SimpleDataset::SimpleImage::get_location() const 
{
	return image_path;
}

std::string SimpleDataset::SimpleImage::get_filename() const 
{
	std::string filename;
	size_t strlen = image_path.length();
	size_t pos = image_path.find_last_of("/");
	int offset = 1;
	if(pos ==-1)
		pos = image_path.find_last_of("\\");

	if(-1 == pos)
	{
		pos = 0;
		offset = 0;
	}

	filename = image_path.substr(pos+offset);
	return filename;
}

adtmath::sparse_vector_t SimpleDataset::load_bow_feature(uint64_t id) const 
{
	return load_bow_feature_cache(id);
}

std::vector<float> SimpleDataset::load_vec_feature(uint64_t id) const 
{
	return load_vec_feature_cache(id);
}

adtmath::sparse_vector_t SimpleDataset::load_bow_feature_cache(uint64_t id) const 
{
	adtmath::sparse_vector_t bow_descriptors;
	uint32_t level0 = id >> 20;
	uint32_t level1 = (id - (level0 << 20)) >> 10;
	std::stringstream ss;
	ss <<  this->location() << "/feats/" <<feat_type << "/bow_descriptors" << "/" << 
	std::setw(4) << std::setfill('0') << level0 << "/" <<
	std::setw(4) << std::setfill('0') << level1 << "/" <<
	std::setw(9) << std::setfill('0') << id << "." << "bow_descriptors";
	std::string location = ss.str();
	// INFO("couldnt find " << location << ;
	if (!adtfilesystem::file_exists(location)) 
		return bow_descriptors;	

	adtfilesystem::load_sparse_vector(location, bow_descriptors);
	return bow_descriptors;
}

std::vector<float> SimpleDataset::load_vec_feature_cache(uint64_t id) const 
{
	std::vector<float> vec_feature;
	uint32_t level0 = id >> 20;
	uint32_t level1 = (id - (level0 << 20)) >> 10;

	std::stringstream ss;
	
	ss <<  this->location() << "/feats/" << "datavec" << "/" << 
	std::setw(4) << std::setfill('0') << level0 << "/" <<
	std::setw(4) << std::setfill('0') << level1 << "/" <<
	std::setw(9) << std::setfill('0') << id << "." << "datavec";
	std::string location = ss.str();

	if (!adtfilesystem::file_exists(location)) 
		return vec_feature;	
	
	adtfilesystem::load_vector(location, vec_feature);

	return vec_feature;
}

bool SimpleDataset::add_image(const std::shared_ptr<const ADTImage> &image) 
{
	if (id_image_map.right.find(image->id) != id_image_map.right.end()) 
		return false;

	const std::shared_ptr<const SimpleDataset::SimpleImage> simage = std::static_pointer_cast<const SimpleDataset::SimpleImage>(image);
	id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type(simage->get_location(), simage->id));

	return true;
}

int64_t SimpleDataset::add_image(const std::string &image_path) 
{
	int64_t image_id_cnt_ret;
	std::string filename;

	int offset =1;
	size_t strlen = image_path.length();
	size_t pos = image_path.find_last_of("/");

	if(pos ==-1)
		pos = image_path.find_last_of("\\");

	if(-1 == pos)
	{
		pos = 0;
		offset = 0;
	}
	

	filename = image_path.substr(pos+offset);
	std::string dest_path_full = image_location(filename);
	bool status = adtfilesystem::copy_file(image_path,dest_path_full);

	if(status)
	{
		image_id_cnt_ret = image_id_cnt;
		id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type(dest_path_full, image_id_cnt));
		image_id_cnt++;
	}
	else
	{
		image_id_cnt_ret = -2;
	}

	return (image_id_cnt_ret);
}

int64_t SimpleDataset::delete_image(const std::string imagefilename)
{
	std::string image_name = image_location(imagefilename);

	id_image_bimap_type::left_const_iterator iter = id_image_map.left.find(image_name) ;
	
	if (iter == id_image_map.left.end()) 
		return -1;
	 
	int64_t imageid = id_image_map.left.at(image_name);

	id_image_map.left.erase(image_name);
	adtfilesystem::delete_file(image_name);
	return imageid;
}

