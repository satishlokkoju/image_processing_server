/**********************************************************************************
 * Image Processing Engine :
 * Author: Satish Lokkoju
 **********************************************************************************/

#include "ADTAddImageUser.hpp"
#include "framework/ScopedTimer.hpp"
#include "../ADTRetrieval.hpp"

#include <iomanip>
#include <set>

namespace cloudcv
{
    class ADTAddImageUser  : public Algorithm
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
            static const char * name() { return "imageid"; };
            typedef std::string type;
        };


        void process(const std::map<std::string, ParameterBindingPtr>& inArgs,
		     const std::map<std::string, ParameterBindingPtr>& outArgs) override
        {
            TRACE_FUNCTION;
            const std::string usernamestr = getInput<emailid>(inArgs);
            const std::string fullimagepath = getInput<imagepath>(inArgs);
            std::string &outputstr= getOutput<outputjson>(outArgs);
	    std::string descType = "AKAZE";
	    std::stringstream ss("");
	    adtRetrievalengine =std::make_shared<ADTRetrieval>(VOCABULARYTREE,descType,UKBENCHLOCATION,USERS_LOCATION);
	    int64_t imgid = adtRetrievalengine->addImageUser(usernamestr,fullimagepath);
	    if(imgid >=0)
	    	ss << imgid;
	    else if(imgid == -1)
		ss << "user with given email id is not present";
	    else if(imgid == -2)
		ss << "Image already exists";
		 
	    outputstr = ss.str();
            LOG_TRACE_MESSAGE(usernamestr);
        }
    protected:
    private:
	std::shared_ptr<ADTRetrieval> adtRetrievalengine;
    };

    ADTAddImageUserInfo::ADTAddImageUserInfo() : AlgorithmInfo("addimageuser_emailid",
        {
            { inputArgument<ADTAddImageUser::emailid>() },
            { inputArgument<ADTAddImageUser::imagepath>() }
        },
        {
            { outputArgument<ADTAddImageUser::outputjson>() }
        }
        )
    {
    }
    
    AlgorithmPtr ADTAddImageUserInfo::create() const
    {
        return AlgorithmPtr(new ADTAddImageUser());
    }

}
