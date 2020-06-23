#include "framework/Algorithm.hpp"
#include "framework/Logger.hpp"
#include "framework/ScopedTimer.hpp"
#include "framework/Job.hpp"
#include "framework/marshal/marshal.hpp"
//#include "framework/NanCheck.hpp"

#include <node.h>
#include <v8.h>
#include <nan.h>

namespace cloudcv
{
    class AlgorithmTask : public Job
    {
        AlgorithmPtr m_algorithm;
        std::map<std::string, ParameterBindingPtr> m_input;
        std::map<std::string, ParameterBindingPtr> m_output;

    public:

        AlgorithmTask( AlgorithmPtr alg, 
            std::map<std::string, ParameterBindingPtr> inArgs,
            std::map<std::string, ParameterBindingPtr> outArgs,
            Nan::Callback * callback) : Job(callback) , m_algorithm(alg), m_input(inArgs) , m_output(outArgs)
        {
            TRACE_FUNCTION;
            LOG_TRACE_MESSAGE("Input arguments:" << inArgs.size());
            LOG_TRACE_MESSAGE("Output arguments:" << outArgs.size());
        }

    protected:

        // This function is executed in another thread at some point after it has been
        // scheduled. IT MUST NOT USE ANY V8 FUNCTIONALITY. Otherwise your extension
        // will crash randomly and you'll have a lot of fun debugging.
        // If you want to use parameters passed into the original call, you have to
        // convert them to PODs or some other fancy method.
        void ExecuteNativeCode() override
        {
            try
            {
                TRACE_FUNCTION;
                m_algorithm->process(m_input, m_output);
            }
            catch (ArgumentException& err)
            {
                LOG_TRACE_MESSAGE("ArgumentException:" << err.what());
                SetErrorMessage(err.what());
            }
            catch (cv::Exception& err)
            {
                LOG_TRACE_MESSAGE("cv::Exception:" << err.what());
                SetErrorMessage(err.what());
            }
            catch (std::runtime_error& err)
            {
                LOG_TRACE_MESSAGE("std::runtime_error:" << err.what());
                SetErrorMessage(err.what());
            }
        }

        // This function is executed in the main V8/JavaScript thread. That means it's
        // safe to use V8 functions again. Don't forget the HandleScope!
        v8::Local<v8::Value> CreateCallbackResult() override
        {
            TRACE_FUNCTION;            

            Nan::EscapableHandleScope scope;

            v8::Local<v8::Object> outputArgument = Nan::New<v8::Object>();

            for (const auto& arg : m_output)
            {
                outputArgument->Set(Nan::Marshal(arg.first), arg.second->marshalFromNative());
            }

            return scope.Escape(outputArgument);
        }
    };


    void ProcessAlgorithm(AlgorithmInfoPtr algorithm, v8::Local<v8::Object> inputArguments, v8::Local<v8::Function> resultsCallback)
    {
        TRACE_FUNCTION;
        
        using namespace cloudcv;

        TRACE_FUNCTION;
        Nan::HandleScope scope;

        try
        {
            //Nan::TryCatch trycatch;

            auto info = algorithm;
            std::map<std::string, ParameterBindingPtr> inArgs, outArgs;

            for (auto arg : info->inputArguments())
            {
                auto propertyName = Nan::Marshal(arg.first);
                v8::Local<v8::Value> argumentValue = Nan::Null();

                if (inputArguments->HasRealNamedProperty(propertyName->ToString()))
                    argumentValue = inputArguments->Get(propertyName);

                LOG_TRACE_MESSAGE("Binding input argument " << arg.first);
                auto bind = arg.second->bind(argumentValue);

                inArgs.insert(std::make_pair(arg.first, bind));
            }

            for (auto arg : info->outputArguments())
            {
                LOG_TRACE_MESSAGE("Binding output argument " << arg.first);
                auto bind = arg.second->bind();
                outArgs.insert(std::make_pair(arg.first, bind));
            }

            //if (trycatch.HasCaught())
            //{
            //    //auto msg = marshal<std::string>(trycatch.Message()->Get());
            //    //LOG_TRACE_MESSAGE(msg);
            //}

            //if (trycatch.CanContinue())
            {
                Nan::Callback * callback = new Nan::Callback(resultsCallback);
                Nan::AsyncQueueWorker(new AlgorithmTask(algorithm->create(), inArgs, outArgs, callback));
            }
        }
        catch (cv::Exception& er)
        {
            LOG_TRACE_MESSAGE(er.what());
            std::string error = er.what();
            v8::Local<v8::Value> argv[] = { Nan::Marshal(error), Nan::Null() };
            Nan::Callback(resultsCallback).Call(2, argv);
        }
        catch (ArgumentException& er)
        {
            Nan::EscapableHandleScope scope;
            LOG_TRACE_MESSAGE(er.what());
            std::string error = er.what();
            v8::Local<v8::Value> argv[] = { scope.Escape(Nan::Marshal(error)), Nan::Null() };
            Nan::Callback(resultsCallback).Call(2, (argv));
        }
        catch (std::runtime_error& er)
        {
            LOG_TRACE_MESSAGE(er.what());
            std::string error = er.what();
            v8::Local<v8::Value> argv[] = { Nan::Marshal(error), Nan::Null() };
            Nan::Callback(resultsCallback).Call(2, argv);
        }
        catch (...)
        {
            LOG_TRACE_MESSAGE("Unknown error");
        }
    }
}
