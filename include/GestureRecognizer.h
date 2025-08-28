// GestureRecognizer.h

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

enum class GestureType {
    NONE,
    ARMS_CROSSED,
    ONE_ARM_UP,
    PEACE,
    THUMBS_UP,
    THUMBS_DOWN
};

struct Keypoint {
    cv::Point2f point;
    float confidence;
};

struct RecognitionResult {
    GestureType finalGesture = GestureType::NONE;
    std::vector<Keypoint> poseKeypoints;
    std::vector<Keypoint> leftHandKeypoints;
    std::vector<Keypoint> rightHandKeypoints;
};

namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
}

class GestureRecognizer {
public:
    GestureRecognizer();
    ~GestureRecognizer();

    void loadBodyPoseModel(const std::string& path);
    void loadHandPoseModel(const std::string& path);
    void loadClassifierModel(const std::string& path);

    RecognitionResult recognize(const cv::Mat& personFrame);

private:
    std::unique_ptr<Ort::Env> m_bodyPoseEnv;
    std::unique_ptr<Ort::Session> m_bodyPoseSession;
    std::unique_ptr<Ort::SessionOptions> m_bodyPoseSessionOptions;

    std::unique_ptr<Ort::Env> m_handPoseEnv;
    std::unique_ptr<Ort::Session> m_handPoseSession;
    std::unique_ptr<Ort::SessionOptions> m_handPoseSessionOptions;

    std::unique_ptr<Ort::Env> m_classifierEnv;
    std::unique_ptr<Ort::Session> m_classifierSession;
    std::unique_ptr<Ort::SessionOptions> m_classifierSessionOptions;
    
    std::vector<GestureType> m_classMap;
};