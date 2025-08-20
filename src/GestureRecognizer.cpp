// GestureRecognizer.cpp

#include "GestureRecognizer.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <json.hpp>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <filesystem>


struct Keypoint {
    cv::Point2f point;
    float confidence;
};

enum COCO_PARTS {
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10
};

const int POSE_INPUT_WIDTH = 640;
const int POSE_INPUT_HEIGHT = 640;
const float POSE_CONFIDENCE_THRESHOLD = 0.5f;


GestureRecognizer::GestureRecognizer(std::shared_ptr<HttpClient> httpClient) : m_httpClient(httpClient) {
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "gesture-recognizer");
    m_sessionOptions = std::make_unique<Ort::SessionOptions>();
}

GestureRecognizer::~GestureRecognizer() = default;

void GestureRecognizer::loadModel(const std::string& path) {
    m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    m_session = std::make_unique<Ort::Session>(*m_env, path.c_str(), *m_sessionOptions);
    std::cout << "Loading model from " << path << std::endl;
}


GestureType analyzeKeypoints(const std::vector<Keypoint>& keypoints){
    if (keypoints.size() < 17){
        return GestureType::NONE;
    }

    const float min_confidence = 0.5f;

    // Одна рука поднята
    bool left_arm_up = keypoints[LEFT_WRIST].point.y < keypoints[LEFT_SHOULDER].point.y && 
        keypoints[LEFT_WRIST].confidence > min_confidence && 
        keypoints[LEFT_SHOULDER].confidence > min_confidence;
    bool right_arm_up = keypoints[RIGHT_WRIST].point.y < keypoints[RIGHT_SHOULDER].point.y && 
        keypoints[RIGHT_WRIST].confidence > min_confidence && 
        keypoints[RIGHT_SHOULDER].confidence > min_confidence;

    if (left_arm_up || right_arm_up) {
        return GestureType::ONE_ARM_UP;
    }

    // Руки скрещены
    bool arms_crossed = keypoints[LEFT_WRIST].point.x > keypoints[RIGHT_SHOULDER].point.x &&
        keypoints[RIGHT_WRIST].point.x < keypoints[LEFT_SHOULDER].point.x && 
        keypoints[LEFT_WRIST].confidence > min_confidence &&
        keypoints[RIGHT_WRIST].confidence > min_confidence &&
        keypoints[LEFT_SHOULDER].confidence > min_confidence &&
        keypoints[RIGHT_SHOULDER].confidence > min_confidence; 

    if (arms_crossed) {
        return GestureType::ARMS_CROSSED;
    }

    return GestureType::NONE;
}

GestureType GestureRecognizer::recognize(const cv::Mat& frameWithPerson) {
    std::vector<unsigned char> jpegBuffer;
    cv::imencode(".jpg", frameWithPerson, jpegBuffer);

    std::string json_response = m_httpClient->sendHandData(jpegBuffer);
    
    static const std::map<std::string, GestureType> gestureMap = {
        {"NONE", GestureType::NONE},
        {"ARMS_CROSSED", GestureType::ARMS_CROSSED},
        {"ONE_ARM_UP", GestureType::ONE_ARM_UP},
        {"PEACE", GestureType::PEACE},
        {"THUMBS_UP", GestureType::THUMBS_UP},
        {"THUMBS_DOWN", GestureType::THUMBS_DOWN}
    };

    try {
        auto json = nlohmann::json::parse(json_response);
        std::string gesture_str = json.at("gesture");
        auto it = gestureMap.find(gesture_str);
        if (it != gestureMap.end()) {
            return it->second;
        }
    } catch(const std::exception& e) {
        
    }
    
    return GestureType::NONE;
}