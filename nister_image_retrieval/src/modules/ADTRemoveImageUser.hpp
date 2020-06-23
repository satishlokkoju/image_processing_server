/**********************************************************************************
 * Image Retrieval Engine - Native code used to bind Node.js with Image retrieval engine.
 * 
 * Author: satish lokkoju <satish.l@google.com>
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

    class ADTRemoveImageUserInfo : public AlgorithmInfo
    {
    public:
        ADTRemoveImageUserInfo();

        AlgorithmPtr create() const override;
    };

}
