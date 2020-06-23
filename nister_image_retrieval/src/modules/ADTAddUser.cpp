/**********************************************************************************
 * Image Processing Engine :
 * Author: Satish Lokkoju
 **********************************************************************************/

#include "ADTAddUser.hpp"
#include "framework/ScopedTimer.hpp"
#include "../ADTRetrieval.hpp"

#include <iomanip>
#include <set>

namespace cloudcv
{
    class ADTAddUser  : public Algorithm
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
	    bool retval = adtRetrievalengine->addUser(usernamestr);
	    if(retval)
	    	outputstr = "User added successfully";
	    else
		outputstr = "User already exists!";

            LOG_TRACE_MESSAGE(usernamestr);
        }
    protected:
    private:
	std::shared_ptr<ADTRetrieval> adtRetrievalengine;
    };

    ADTAddUserInfo::ADTAddUserInfo() : AlgorithmInfo("adduser_emailid",
        {
            { inputArgument<ADTAddUser::emailid>() }
        },
        {
            { outputArgument<ADTAddUser::outputjson>() }
        }
        )
    {
    }
    
    AlgorithmPtr ADTAddUserInfo::create() const
    {
        return AlgorithmPtr(new ADTAddUser());
    }

}
