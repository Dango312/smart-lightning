#include "GestureRecognizer.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include "spdlog/spdlog.h"
#include "ConfigManager.h"

// Параметры модели
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.5f;

enum BodyParts {
    LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6, LEFT_ELBOW = 7, RIGHT_ELBOW = 8,
    LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_HIP = 11, RIGHT_HIP = 12
};

std::vector<float> normalizeYoloLandmarks(
    std::vector<Keypoint>& pose_kps_in, 
    std::vector<Keypoint>& lh_kps_in, 
    std::vector<Keypoint>& rh_kps_in) 
{
    auto pose_kps = pose_kps_in;
    auto lh_kps = lh_kps_in;
    auto rh_kps = rh_kps_in;

    // Нормализация позы
    if (pose_kps.size() == 17) {
        cv::Point2f shoulders = (pose_kps[5].point + pose_kps[6].point) * 0.5f;
        cv::Point2f hips = (pose_kps[11].point + pose_kps[12].point) * 0.5f;
        float torso = cv::norm(shoulders - hips);
        if (torso > 0.01) {
            for(auto& kp : pose_kps) { 
                if (kp.confidence > 0.0) {
                    kp.point = (kp.point - shoulders) / torso; 
                }
            }
        }
    }

    // Нормализация левой руки
    if (lh_kps.size() == 21 && cv::norm(lh_kps[0].point) > 0.0) {
        cv::Point2f wrist = lh_kps[0].point;
        std::vector<cv::Point2f> relative_kps;
        relative_kps.reserve(21);
        for(const auto& kp : lh_kps) {
            relative_kps.push_back(kp.point - wrist);
        }

        float max_dist = 0.0f;
        for(const auto& pt : relative_kps) {
            max_dist = std::max(max_dist, (float)cv::norm(pt));
        }

        if (max_dist > 0) {
            for(size_t i = 0; i < lh_kps.size(); ++i) {
                lh_kps[i].point = relative_kps[i] / max_dist;
            }
        } else {
            for(auto& kp : lh_kps) { kp.point = cv::Point2f(0,0); }
        }
    }

    // Нормализация правой руки
    if (rh_kps.size() == 21 && cv::norm(rh_kps[0].point) > 0.0) {
        cv::Point2f wrist = rh_kps[0].point;
        std::vector<cv::Point2f> relative_kps;
        relative_kps.reserve(21);
        for(const auto& kp : rh_kps) {
            relative_kps.push_back(kp.point - wrist);
        }
        float max_dist = 0.0f;
        for(const auto& pt : relative_kps) {
            max_dist = std::max(max_dist, (float)cv::norm(pt));
        }
        if (max_dist > 0) {
            for(size_t i = 0; i < rh_kps.size(); ++i) {
                rh_kps[i].point = relative_kps[i] / max_dist;
            }
        } else {
            for(auto& kp : rh_kps) { kp.point = cv::Point2f(0,0); }
        }
    }
    std::vector<float> result;
    result.reserve(118);
    for(const auto& kp : pose_kps) { result.push_back(kp.point.x); result.push_back(kp.point.y); }
    for(const auto& kp : lh_kps) { result.push_back(kp.point.x); result.push_back(kp.point.y); }
    for(const auto& kp : rh_kps) { result.push_back(kp.point.x); result.push_back(kp.point.y); }
    return result;
}
std::vector<Keypoint> extractKeypointsFromModel(
    const Ort::Value& tensor, int num_keypoints, float x_factor, float y_factor, float roi_x = 0, float roi_y = 0) 
{
    std::vector<Keypoint> keypoints(num_keypoints, {cv::Point2f(0,0), 0.0f});
    const float* raw_output = tensor.GetTensorData<float>();
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    int num_proposals = shape[2];
    
    if (num_proposals == 0) return keypoints;

    int best_proposal_idx = -1;
    float max_conf = 0.0f;
    for (int i = 0; i < num_proposals; ++i) {
        float conf = raw_output[4 * num_proposals + i];
        if (conf > max_conf) {
            max_conf = conf;
            best_proposal_idx = i;
        }
    }

    if (max_conf < CONFIDENCE_THRESHOLD) return keypoints;

    const float* kps_data = raw_output + 5 * num_proposals + best_proposal_idx;
    for (int i = 0; i < num_keypoints; ++i) {
        float x = kps_data[i * 3 * num_proposals];
        float y = kps_data[(i * 3 + 1) * num_proposals];
        float conf = kps_data[(i * 3 + 2) * num_proposals];
        
        float visible_conf = 1.0f / (1.0f + expf(-conf));
        
        if (visible_conf > 0.5f) {
            keypoints[i] = {
                cv::Point2f((x * x_factor) + roi_x, (y * y_factor) + roi_y),
                visible_conf
            };
        }
    }
    return keypoints;
}

GestureRecognizer::GestureRecognizer() {
    m_bodyPoseEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "body-pose");
    m_bodyPoseSessionOptions = std::make_unique<Ort::SessionOptions>();
    
    m_handPoseEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "hand-pose");
    m_handPoseSessionOptions = std::make_unique<Ort::SessionOptions>();
    
    m_classifierEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "classifier");
    m_classifierSessionOptions = std::make_unique<Ort::SessionOptions>();

    m_classMap = {
        GestureType::ARMS_CROSSED,
        GestureType::NONE,
        GestureType::ONE_ARM_UP,
        GestureType::PEACE,
        GestureType::THUMBS_DOWN,
        GestureType::THUMBS_UP
    };
}

GestureRecognizer::~GestureRecognizer() = default;

void GestureRecognizer::loadBodyPoseModel(const std::string& path) {
    std::string device = ConfigManager::getInstance().getDevice();
    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_options{};
        try { m_bodyPoseSessionOptions->AppendExecutionProvider_CUDA(cuda_options); } 
        catch (const Ort::Exception& e) { spdlog::error("Failed to add CUDA to BodyPoseModel: {}", e.what()); }
    }
    m_bodyPoseSession = std::make_unique<Ort::Session>(*m_bodyPoseEnv, path.c_str(), *m_bodyPoseSessionOptions);
    spdlog::info("GestureRecognizer: Body Pose model loaded from {}", path);
}

void GestureRecognizer::loadHandPoseModel(const std::string& path) {
    std::string device = ConfigManager::getInstance().getDevice();
    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_options{};
        try { m_handPoseSessionOptions->AppendExecutionProvider_CUDA(cuda_options); } 
        catch (const Ort::Exception& e) { spdlog::error("Failed to add CUDA to HandPoseModel: {}", e.what()); }
    }
    m_handPoseSession = std::make_unique<Ort::Session>(*m_handPoseEnv, path.c_str(), *m_handPoseSessionOptions);
    spdlog::info("GestureRecognizer: Hand Pose model loaded from {}", path);
}

void GestureRecognizer::loadClassifierModel(const std::string& path) {
    m_classifierSession = std::make_unique<Ort::Session>(*m_classifierEnv, path.c_str(), *m_classifierSessionOptions);
    spdlog::info("GestureRecognizer: Classifier model loaded from {}", path);
}

RecognitionResult GestureRecognizer::recognize(const cv::Mat& personFrame) {
    RecognitionResult result;
    if (!m_bodyPoseSession || !m_handPoseSession || !m_classifierSession) {
        throw std::runtime_error("GestureRecognizer models not loaded!");
    }
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape{1, 3, INPUT_HEIGHT, INPUT_WIDTH};

    // Анализ позы (17 точек)
    cv::Mat body_blob;
    cv::dnn::blobFromImage(personFrame, body_blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    Ort::Value body_input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)body_blob.data, body_blob.total(), input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto body_pose_tensors = m_bodyPoseSession->Run(Ort::RunOptions{nullptr}, input_names, &body_input_tensor, 1, output_names, 1);
    
    result.poseKeypoints = extractKeypointsFromModel(
        body_pose_tensors[0], 17, 
        personFrame.cols / (float)INPUT_WIDTH, 
        personFrame.rows / (float)INPUT_HEIGHT
    );
    
    std::vector<Keypoint> lh_kps(21, {cv::Point2f(0,0), 0.0f});
    std::vector<Keypoint> rh_kps(21, {cv::Point2f(0,0), 0.0f});

    // Анализ кисти
    bool left_up = result.poseKeypoints.size() == 17 && result.poseKeypoints[LEFT_WRIST].point.y < result.poseKeypoints[LEFT_ELBOW].point.y && result.poseKeypoints[LEFT_WRIST].confidence > 0.5;
    bool right_up = result.poseKeypoints.size() == 17 && result.poseKeypoints[RIGHT_WRIST].point.y < result.poseKeypoints[RIGHT_ELBOW].point.y && result.poseKeypoints[RIGHT_WRIST].confidence > 0.5;
    
    if (left_up || right_up) {
        Keypoint wrist = left_up ? result.poseKeypoints[LEFT_WRIST] : result.poseKeypoints[RIGHT_WRIST];
        Keypoint elbow = left_up ? result.poseKeypoints[LEFT_ELBOW] : result.poseKeypoints[RIGHT_ELBOW];
        
        cv::Point2f forearm_vector = wrist.point - elbow.point;
        float forearm_length = cv::norm(forearm_vector);
        
        if (forearm_length > 30) {
            cv::Point2f hand_center = wrist.point + (forearm_vector / forearm_length) * forearm_length * 0.6f;
            int box_size = static_cast<int>(forearm_length * 2.5f);
            cv::Rect handRoi(hand_center.x - box_size / 2, hand_center.y - box_size / 2, box_size, box_size);
            handRoi &= cv::Rect(0, 0, personFrame.cols, personFrame.rows);

            if (handRoi.width > 20 && handRoi.height > 20) {
                cv::Mat handFrame = personFrame(handRoi);
                cv::Mat hand_blob;
                cv::dnn::blobFromImage(handFrame, hand_blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
                Ort::Value hand_input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)hand_blob.data, hand_blob.total(), input_shape.data(), input_shape.size());
                
                auto hand_pose_tensors = m_handPoseSession->Run(Ort::RunOptions{nullptr}, input_names, &hand_input_tensor, 1, output_names, 1);
                
                std::vector<Keypoint> precise_hand_kps = extractKeypointsFromModel(
                    hand_pose_tensors[0], 21,
                    handFrame.cols / (float)INPUT_WIDTH,
                    handFrame.rows / (float)INPUT_HEIGHT,
                    handRoi.x, handRoi.y
                );
                
                if (left_up) lh_kps = precise_hand_kps;
                else rh_kps = precise_hand_kps;
            }
        }
    }

    result.leftHandKeypoints = lh_kps;
    result.rightHandKeypoints = rh_kps;

    // Классификация
    std::vector<float> normalized_features = normalizeYoloLandmarks(result.poseKeypoints, result.leftHandKeypoints, result.rightHandKeypoints);

    std::vector<int64_t> classifier_input_shape{1, static_cast<long long>(normalized_features.size())};
    Ort::Value classifier_input_tensor = Ort::Value::CreateTensor<float>(memory_info, normalized_features.data(), normalized_features.size(), classifier_input_shape.data(), classifier_input_shape.size());
    
    const char* classifier_input_names[] = {"float_input"};
    const char* classifier_output_names[] = {"label"};
    auto classifier_output = m_classifierSession->Run(Ort::RunOptions{nullptr}, classifier_input_names, &classifier_input_tensor, 1, classifier_output_names, 1);
    
    const int64_t* label_tensor = classifier_output[0].GetTensorData<int64_t>();
    int64_t predicted_index = label_tensor[0];
    
    if (predicted_index >= 0 && predicted_index < m_classMap.size()) {
        result.finalGesture = m_classMap[predicted_index];
    }

    return result;
} 