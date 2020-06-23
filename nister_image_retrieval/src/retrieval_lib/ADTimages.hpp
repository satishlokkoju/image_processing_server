#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Simple class to visualize the results of search using html_pages*/
/*etc.*/

#include <iostream>
#include <stdint.h>

/// Abstract class representing an image.  Implementing classes must provide a way to 
/// load images and construct image paths for loading features.  See tests and benchmarks
/// for example implementations of Image.
class ADTImage 
{
public:
	ADTImage(uint64_t image_id);
	
	uint64_t id; /// All images are assigned a unique id in the dataset.

	virtual ~ADTImage();

	/// Returns the corresponding feature path given a feature name and feature type(ex. "sift").
	virtual std::string feature_path(const std::string &feat_name) const = 0;

	/// Returns the image location relative to the database data directory.
	virtual std::string get_location() const = 0;

	/// Returns the image location relative to the database data directory.
	virtual std::string get_filename() const = 0;

protected:

	// std::function<std::vector<char>(const std::string &feat_name)> load_function;
private:
	
};


