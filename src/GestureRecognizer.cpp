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

GestureType GestureRecognizer::recognize(const cv::Mat& frame, const cv::Rect& humanRoi){
    if (!m_session) {
        throw std::runtime_error("Gesture recognizer model not loaded!");
    }

    cv::Rect paddedRoi = humanRoi;
    paddedRoi.x -= 20;
    paddedRoi.y -= 20;
    paddedRoi.width += 40;
    paddedRoi.height += 40;
    paddedRoi &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat person_frame = frame(paddedRoi);

    cv::Mat blob;
    cv::dnn::blobFromImage(person_frame, blob, 1./255., cv::Size(POSE_INPUT_WIDTH, POSE_INPUT_HEIGHT), cv::Scalar(), true, false);

    size_t input_tensor_size = POSE_INPUT_WIDTH * POSE_INPUT_HEIGHT * 3;
    std::vector<int64_t> input_shape{1, 3, POSE_INPUT_HEIGHT, POSE_INPUT_WIDTH};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const float* raw_output = output_tensors[0].GetTensorData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_proposals = shape[2];

    int best_proposal_idx = -1;
    float max_confidence = 0.0f;
    for (int i = 0; i < num_proposals; ++i) {
        float confidence = raw_output[4 * num_proposals + i];
        if(confidence > max_confidence){
            max_confidence = confidence;
            best_proposal_idx = i;
        }
    }

    if (max_confidence < POSE_CONFIDENCE_THRESHOLD){
        return GestureType::NONE;
    }

    std::vector<Keypoint> keypoints;
    float x_factor = person_frame.cols / (float)POSE_INPUT_WIDTH;
    float y_factor = person_frame.rows / (float)POSE_INPUT_HEIGHT;

    const float* keypoints_data = raw_output + 5 * num_proposals + best_proposal_idx;

    for (int i = 0; i < 17; ++i){
        float x = keypoints_data[i * 3 * num_proposals];
        float y = keypoints_data[(i * 3 + 1) * num_proposals];
        float conf = keypoints_data[(i * 3 + 2) * num_proposals];
        keypoints.push_back({cv::Point2f(x * x_factor, y * y_factor), conf});
    }

    bool left_hand_raised = keypoints[LEFT_WRIST].point.y < keypoints[LEFT_ELBOW].point.y && keypoints[LEFT_WRIST].confidence > 0.5;
    bool right_hand_raised = keypoints[RIGHT_WRIST].point.y < keypoints[RIGHT_ELBOW].point.y && keypoints[RIGHT_WRIST].confidence > 0.5;

    if (left_hand_raised || right_hand_raised) {
        Keypoint wrist = left_hand_raised ? keypoints[LEFT_WRIST] : keypoints[RIGHT_WRIST];
        Keypoint elbow = left_hand_raised ? keypoints[LEFT_ELBOW] : keypoints[RIGHT_ELBOW];
        cv::Point2f forearm_vector = wrist.point - elbow.point;
        float forearm_length = std::sqrt(forearm_vector.x * forearm_vector.x + forearm_vector.y * forearm_vector.y);
        
        if (forearm_length < 30) { 
            return analyzeKeypoints(keypoints);
        }
        cv::Point2f forearm_direction = forearm_vector / forearm_length;
        float shift_factor = 0.6f;
        cv::Point2f hand_center = wrist.point + forearm_direction * forearm_length * shift_factor;
        float size_factor = 2.5f;
        int box_size = static_cast<int>(forearm_length * size_factor);

        cv::Rect handRoi(
            hand_center.x - box_size / 2, 
            hand_center.y - box_size / 2, 
            box_size, 
            box_size
        );

        handRoi &= cv::Rect(0, 0, person_frame.cols, person_frame.rows);

        if (handRoi.width > 0 && handRoi.height > 0){
            cv::Mat handImage = person_frame(handRoi);


            std::vector<unsigned char> jpegBuffer;
            cv::imencode(".jpg", handImage, jpegBuffer);

            std::string json_response = m_httpClient->sendHandData(jpegBuffer);

            try {
                auto json = nlohmann::json::parse(json_response);
                std::string gesture_str = json.at("gesture");
                if (gesture_str == "PEACE") return GestureType::PEACE;
                if (gesture_str == "THUMBS_UP") return GestureType::THUMBS_UP;
                if (gesture_str == "THUMBS_DOWN") return GestureType::THUMBS_DOWN;
            }
            catch(const std::exception& e) {
            
            }
        }
    }

    return analyzeKeypoints(keypoints);
}