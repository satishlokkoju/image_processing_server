/**********************************************************************************
 * Image Processing Engine :
 * Author: Satish Lokkoju
 **********************************************************************************/

#include "ADTRetrieveImageUser.hpp"
#include "framework/ScopedTimer.hpp"
#include "../ADTRetrieval.hpp"

#include <iomanip>
#include <set>

namespace cloudcv
{
    class ADTRetrieveImageUser  : public Algorithm
    {
    public:

	struct emailid
        {
            static const char * name() { return "emailid"; };
            typedef std::string type;
        };

	struct imagepath
        {
            static const char * name() { return "imagepath"; };
            typedef std::string type;
        };
	struct outputjson
        {
            static const char * name() { return "imageids"; };
            typedef std::string type;
        };


        void process(const std::map<std::string, ParameterBindingPtr>& inArgs,
		     const std::map<std::string, ParameterBindingPtr>& outArgs) override
        {
            TRACE_FUNCTION;
            const std::string usernamestr = getInput<emailid>(inArgs);
            std::string &outputstr= getOutput<outputjson>(outArgs);
            std::string imagePathstr= getInput<imagepath>(inArgs);
	    std::stringstream ss("");
	    std::string descType = "AKAZE";
	    adtRetrievalengine =std::make_shared<ADTRetrieval>(VOCABULARYTREE,descType,UKBENCHLOCATION,USERS_LOCATION);
	    cv::Mat queryMat =  cv::imread(imagePathstr);
	    std::vector<std::string> imgfilenames = adtRetrievalengine->retrieveImage(usernamestr,queryMat,3);
	    if(imgfilenames.size() >0)
	    {
		ss << imgfilenames[0];
            }
            else
            {
               // unable to find any images;
               ss << "-1";
            }

	    for(unsigned int i =1; i< imgfilenames.size(); i++)
	    {
		ss << ",";
		ss << imgfilenames[i];
	    }

	    outputstr = ss.str();
            LOG_TRACE_MESSAGE(usernamestr);
        }
    protected:
    private:
	std::shared_ptr<ADTRetrieval> adtRetrievalengine;
    };

    ADTRetrieveImageUserInfo::ADTRetrieveImageUserInfo() : AlgorithmInfo("retrieveimage_emailid",
        {
            { inputArgument<ADTRetrieveImageUser::emailid>() },
            { inputArgument<ADTRetrieveImageUser::imagepath>() }
        },
        {
            { outputArgument<ADTRetrieveImageUser::outputjson>() }
        }
        )
    {
    }
    
    AlgorithmPtr ADTRetrieveImageUserInfo::create() const
    {
        return AlgorithmPtr(new ADTRetrieveImageUser());
    }

}
