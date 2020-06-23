#include "ADTSearchBase.hpp"

ADTSearchBase::ADTSearchBase() { }
ADTSearchBase::ADTSearchBase(const std::string &file_path) { }

ADTSearchBase::~ADTSearchBase() { }

bool ADTSearchBase::train (const std::shared_ptr<ADTDataset> &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const ADTImage > > &examples)
{
	INFO("Derived class has to implement the training module");
	exit(1);
}