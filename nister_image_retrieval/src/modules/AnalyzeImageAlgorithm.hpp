/**********************************************************************************
 * CloudCV Boostrap - A starter template for Node.js with OpenCV bindings.
 *                    This project lets you to quickly prototype a REST API
 *                    in a Node.js for a image processing service written in C++. 
 * 
 * Author: Eugene Khvedchenya <ekhvedchenya@gmail.com>
 * 
 * More information:
 *  - https://cloudcv.io
 *  - http://computer-vision-talks.com
 * 
 **********************************************************************************/

#pragma once

#include "framework/marshal/marshal.hpp"
#include "framework/Algorithm.hpp"
#include "framework/marshal/opencv.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <opencv2/opencv.hpp>
#include <array>

namespace cloudcv
{

    class AnalyzeAlgorithmInfo : public AlgorithmInfo
    {
    public:
        AnalyzeAlgorithmInfo();

        AlgorithmPtr create() const override;
    };

}
