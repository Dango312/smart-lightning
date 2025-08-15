// HumanDetector.h
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
}

class HumanDetector {
    public:
        HumanDetector();
        ~HumanDetector();

        void loadModel(const std::string& path);

        std::vector<cv::Rect> detect(const cv::Mat& frame);

    private:
        std::unique_ptr<Ort::Env> m_env;
        std::unique_ptr<Ort::Session> m_session;
        std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
        std::chrono::steady_clock::time_point m_startTime;
};