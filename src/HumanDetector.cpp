// HumanDetector.cpp

#include "HumanDetector.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include "spdlog/spdlog.h"

// --- Параметры модели ---
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.45f;
const float CONFIDENCE_THRESHOLD = 0.45f;


HumanDetector::HumanDetector() {
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "human-detector");
    m_sessionOptions = std::make_unique<Ort::SessionOptions>();
}

HumanDetector::~HumanDetector() = default;

void HumanDetector::loadModel(const std::string& path) {
    m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    m_session = std::make_unique<Ort::Session>(*m_env, path.c_str(), *m_sessionOptions);

    spdlog::info("HumanDetector: loaded model from {}", path);
}

std::vector<cv::Rect> HumanDetector::detect(const cv::Mat& frame) {
    if (!m_session) {
        throw std::runtime_error("HumanDetector model not loaded!");
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    size_t input_tensor_size = INPUT_WIDTH * INPUT_HEIGHT * 3;
    std::vector<int64_t> input_shape{1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const float* raw_output = output_tensors[0].GetTensorData<float>();
    
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_proposals = shape[2]; 
    int proposal_length = shape[1];

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = frame.cols / (float)INPUT_WIDTH;
    float y_factor = frame.rows / (float)INPUT_HEIGHT;

    for (int i = 0; i < num_proposals; ++i) {
        const float* proposal_data = raw_output + i;
        
        const float* class_scores = proposal_data + 4 * num_proposals;

        cv::Mat scores(80, 1, CV_32F, (void*)class_scores, num_proposals * sizeof(float));
        cv::Point class_id_point;
        double max_score;
        cv::minMaxLoc(scores, 0, &max_score, 0, &class_id_point);

        if (max_score > SCORE_THRESHOLD) {
            if (class_id_point.y == 0) { 
                confidences.push_back(max_score);
                class_ids.push_back(class_id_point.y);

                float cx = proposal_data[0 * num_proposals];
                float cy = proposal_data[1 * num_proposals];
                float w = proposal_data[2 * num_proposals];
                float h = proposal_data[3 * num_proposals];

                int left = static_cast<int>((cx - 0.5 * w) * x_factor);
                int top = static_cast<int>((cy - 0.5 * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    
    std::vector<cv::Rect> final_boxes;
    for (int idx : nms_result) {
        final_boxes.push_back(boxes[idx]);
    }

    return final_boxes;
}