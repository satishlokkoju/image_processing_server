#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Simple class to visualize the results of search using html_pages*/
/*etc.*/

#include "ADTDataset.hpp"
#include <string>

class ADTMatchesPage 
{
public: 

	ADTMatchesPage(uint32_t max_matches_per_page = 16, uint32_t max_images_per_match = 16);
	~ADTMatchesPage();

	void add_match(std::shared_ptr<const SimpleDataset::SimpleImage> query_image , std::vector<uint64_t> &match_ids, const std::shared_ptr<ADTDataset> &dataset,std::shared_ptr< std::vector<int> > validated = std::shared_ptr< std::vector<int> >());

	/// Writes out all the html match mages to the input specified folder.  The first page 
	/// will look something like folder/matches_00000.html.
	void write(const std::string &folder) const;

protected:

	std::string stylesheet() const;  /// Returns a string containing css stylesheet
	std::string header() const; /// Returns a string containing the html header
	std::string footer() const; /// Returns a string containing the html footer
	std::string navbar(uint32_t cur_page, uint32_t max_pages) const; /// Returns a string containing the navbar needed for pagination
	std::string pagename(uint32_t cur_page) const; /// Returns a string containing the pagename (ex. matches_00001.html)

	std::vector<std::string> html_strings; /// Holds the html strings for each match passed into add_match

	uint32_t max_matches_per_page_, max_images_per_match_;

};