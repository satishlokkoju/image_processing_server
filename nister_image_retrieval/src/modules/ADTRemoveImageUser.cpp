/**********************************************************************************
 * Image Processing Engine :
 * Author: Satish Lokkoju
 **********************************************************************************/

#include "ADTRemoveImageUser.hpp"
#include "framework/ScopedTimer.hpp"
#include "../ADTRetrieval.hpp"

#include <iomanip>
#include <set>

namespace cloudcv
{
    class ADTRemoveImageUser  : public Algorithm
    {
    public:

	struct emailid
        {
            static const char * name() { return "emailid"; };
            typedef std::string type;
        };

	struct imagename
        {
            static const char * name() { return "imagename"; };
            typedef std::string type;
        };

	struct outputjson
        {
            static const char * name() { return "outputjson"; };
            typedef std::string type;
        };


        void process(const std::map<std::string, ParameterBindingPtr>& inArgs,
		     const std::map<std::string, ParameterBindingPtr>& outArgs) override
        {
            TRACE_FUNCTION;
            const std::string usernamestr = getInput<emailid>(inArgs);
            const std::string imagenameval = getInput<imagename>(inArgs);
            std::string &outputstr= getOutput<outputjson>(outArgs);
	    std::string descType = "AKAZE";
	    adtRetrievalengine =std::make_shared<ADTRetrieval>(VOCABULARYTREE,descType,UKBENCHLOCATION,USERS_LOCATION);
	    bool status = adtRetrievalengine->removeImageUser(usernamestr,imagenameval);
	    
	    if(status)
	    	outputstr = "deleted image from the user database";
	    else
		outputstr =  "unable to delete the image from the user";

            LOG_TRACE_MESSAGE(usernamestr);
        }
    protected:
    private:
	std::shared_ptr<ADTRetrieval> adtRetrievalengine;
    };

    ADTRemoveImageUserInfo::ADTRemoveImageUserInfo() : AlgorithmInfo("removeimageuser_emailid",
        {
            { inputArgument<ADTRemoveImageUser::emailid>() },
            { inputArgument<ADTRemoveImageUser::imagename>() }
        },
        {
            { outputArgument<ADTRemoveImageUser::outputjson>() }
        }
        )
    {
    }
    
    AlgorithmPtr ADTRemoveImageUserInfo::create() const
    {
        return AlgorithmPtr(new ADTRemoveImageUser());
    }

}
