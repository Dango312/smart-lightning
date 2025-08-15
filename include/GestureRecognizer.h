// GestureRecognizer.h

#pragma once 

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

enum class GestureType {
    NONE,
    ARMS_CROSSED,
    ONE_ARM_UP,
    CUSTOM_1,
    CUSTOM_2,
    CUSTOM_3
};

class GestureRecognizer {
    public:
        GestureRecognizer();

        void loadModel(const std::string& path);

        GestureType recognize(const cv::Mat& frame, const cv::Rect& humanRoi);
    
    private:
        std::chrono::steady_clock::time_point m_startTime;
};
