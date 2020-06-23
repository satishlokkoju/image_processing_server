/**********************************************************************************
 * Image Processing Engine
 * Author: Satish Lokkoju
 **********************************************************************************/

#include "AnalyzeImageAlgorithm.hpp"
#include "framework/ScopedTimer.hpp"

#include <iomanip>
#include <set>

namespace cloudcv
{
    class testCloudCV  : public Algorithm
    {
    public:

	struct username
        {
            static const char * name() { return "username"; };
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
            const std::string usernamestr = getInput<username>(inArgs);
            std::string &outputstr= getOutput<outputjson>(outArgs);
	    outputstr = "json output as requested";
            LOG_TRACE_MESSAGE(usernamestr);
        }
    protected:
    private:

    };

    AnalyzeAlgorithmInfo::AnalyzeAlgorithmInfo() : AlgorithmInfo("testingCloudCV",
        {
            { inputArgument<testCloudCV::username>() }
        },
        {
            { outputArgument<testCloudCV::outputjson>() }
        }
        )
    {
    }
    
    AlgorithmPtr AnalyzeAlgorithmInfo::create() const
    {
        return AlgorithmPtr(new testCloudCV());
    }

}
