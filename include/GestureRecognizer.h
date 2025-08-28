// GestureRecognizer.h

#pragma once 

#include <opencv2/opencv.hpp>
#include "HttpClient.h"
#include <string>
#include <chrono>
#include <memory>

enum class GestureType {
    NONE,
    ARMS_CROSSED,
    ONE_ARM_UP,
    PEACE,
    THUMBS_UP,
    THUMBS_DOWN
};

namespace Ort{
    struct Env;
    struct Session;
    struct SessionOptions;
}

class PythonGestureRecognizer;

class GestureRecognizer {
    public:
        GestureRecognizer();
        ~GestureRecognizer();

        GestureType recognize(const cv::Mat& frameWithPerson);
    
    private:
        std::unique_ptr<PythonGestureRecognizer> m_pyRecognizer;

        std::filesystem::path m_debugDir;
};
