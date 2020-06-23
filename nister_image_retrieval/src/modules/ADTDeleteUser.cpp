/**********************************************************************************
 * Image Processing Engine :
 * Author: Satish Lokkoju
 **********************************************************************************/

#include "ADTDeleteUser.hpp"
#include "framework/ScopedTimer.hpp"
#include "../ADTRetrieval.hpp"

#include <iomanip>
#include <set>

namespace cloudcv
{
    class ADTDeleteUser  : public Algorithm
    {
    public:

	struct emailid
        {
            static const char * name() { return "emailid"; };
            typedef std::string type;
        };
	struct outputjson
        {
            static const char * name() { return "status"; };
            typedef std::string type;
        };


        void process(const std::map<std::string, ParameterBindingPtr>& inArgs,
		     const std::map<std::string, ParameterBindingPtr>& outArgs) override
        {
            TRACE_FUNCTION;
            const std::string usernamestr = getInput<emailid>(inArgs);
            std::string &outputstr= getOutput<outputjson>(outArgs);
	    std::string descType = "AKAZE";
	    adtRetrievalengine =std::make_shared<ADTRetrieval>(VOCABULARYTREE,descType,UKBENCHLOCATION,USERS_LOCATION);
            bool retval	= adtRetrievalengine->deletUser(usernamestr);
	    if(retval)
	    	outputstr = "User deleted successfully";
	    else
		outputstr = "User is not listed!";

            LOG_TRACE_MESSAGE(usernamestr);
        }
    protected:
    private:
	std::shared_ptr<ADTRetrieval> adtRetrievalengine;
    };

    ADTDeleteUserInfo::ADTDeleteUserInfo() : AlgorithmInfo("deleteuser_emailid",
        {
            { inputArgument<ADTDeleteUser::emailid>() }
        },
        {
            { outputArgument<ADTDeleteUser::outputjson>() }
        }
        )
    {
    }
    
    AlgorithmPtr ADTDeleteUserInfo::create() const
    {
        return AlgorithmPtr(new ADTDeleteUser());
    }

}
