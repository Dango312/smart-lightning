// GestureRecognizer.cpp

#include "GestureRecognizer.h"
#include <iostream>

GestureRecognizer::GestureRecognizer() {
    m_startTime = std::chrono::steady_clock::now();
}

void GestureRecognizer::loadModel(const std::string& path) {
    std::cout << "[STUB] GestureRecognizer: Simulating loading model from " << path << std::endl;
}

GestureType GestureRecognizer::recognize(const cv::Mat& frame, const cv::Rect& humanRoi){
    auto now = std::chrono::steady_clock::now();
    long long elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(now - m_startTime).count();

    int state = 0;//(elapsedSeconds / 5) % 2;

    GestureType gesture;
    switch (state) {
        case 0:
            gesture = GestureType::NONE;
            break;
        case 1:
            gesture = GestureType::ARMS_CROSSED;
            break;
        case 2:
            gesture = GestureType::ONE_ARM_UP;
            break;
        case 3:
            gesture = GestureType::CUSTOM_1;
            break;
        default:
            gesture = GestureType::NONE;
            break;
    }
    //std::cout << "Detected gesture" << static_cast<int>(gesture) << std::endl;
    return gesture;
}