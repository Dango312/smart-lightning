// src/GestureRecognizer.cpp 
#include "GestureRecognizer.h"
#include "wrapper.h"
#include <map>
#include <opencv2/opencv.hpp>
#include <pybind11/gil.h>


GestureRecognizer::GestureRecognizer() {
    m_pyRecognizer = std::make_unique<PythonGestureRecognizer>();
}

GestureRecognizer::~GestureRecognizer() = default;

GestureType GestureRecognizer::recognize(const cv::Mat& frameWithPerson) {
    if (frameWithPerson.empty()) {
        return GestureType::NONE;
    }
    std::vector<unsigned char> jpegBuffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", frameWithPerson, jpegBuffer, params);

    std::string gesture_str = m_pyRecognizer->recognize(jpegBuffer);
    
    static const std::map<std::string, GestureType> gestureMap = {
        {"NONE", GestureType::NONE},
        {"ARMS_CROSSED", GestureType::ARMS_CROSSED},
        {"ONE_ARM_UP", GestureType::ONE_ARM_UP},
        {"PEACE", GestureType::PEACE},
        {"THUMBS_UP", GestureType::THUMBS_UP},
        {"THUMBS_DOWN", GestureType::THUMBS_DOWN}
    };

    auto it = gestureMap.find(gesture_str);
    return (it != gestureMap.end()) ? it->second : GestureType::NONE;
}