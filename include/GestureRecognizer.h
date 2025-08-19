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

class GestureRecognizer {
    public:
        GestureRecognizer(std::shared_ptr<HttpClient> htttpClient);
        ~GestureRecognizer();

        void loadModel(const std::string& path);

        GestureType recognize(const cv::Mat& frame, const cv::Rect& humanRoi);
    
    private:
        std::filesystem::path m_debugDir;

        std::shared_ptr<HttpClient> m_httpClient;
        std::unique_ptr<Ort::Env> m_env;
        std::unique_ptr<Ort::Session> m_session;
        std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
};
