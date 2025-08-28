// CameraProcessor.cpp

#include "CameraProcessor.h"
#include "spdlog/spdlog.h"
#include <iostream>

CameraProcessor::CameraProcessor(
    const CameraConfig& config,
    std::shared_ptr<SystemState> systemState,
    std::shared_ptr<HttpClient> httpClient,
    std::shared_ptr<HumanDetector> humanDetector,
    std::shared_ptr<GestureRecognizer> gestureRecognizer)
    : m_config(config),
        m_systemState(systemState),
        m_httpClient(httpClient),
        m_humanDetector(humanDetector),
        m_gestureRecognizer(gestureRecognizer),
        m_isRunning(true),
        m_lastDetectedGesture(GestureType::NONE),
        m_gestureCounter(0) {

    
    int cooldownSec=5;

    m_cooldownDuration = std::chrono::seconds(cooldownSec);
    m_lastRequestTime = std::chrono::steady_clock::now() - m_cooldownDuration;
}

void CameraProcessor::stop(){
    m_isRunning.store(false);
}

cv::Mat CameraProcessor::getLatestFrame() {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    return m_latestFrame.clone();
}

void CameraProcessor::run(){
    spdlog::info("Starting processor for camera ID: {}", m_config.id);

    // Открытие камеры    
    cv::VideoCapture cap;
    try {
        int cameraIndex = std::stoi(m_config.videoUrl);
        cap.open(cameraIndex);
    }
    catch (const std::invalid_argument&){
        cap.open(m_config.videoUrl);
    }
    spdlog::info("Opened camera: {}", m_config.videoUrl);

    if (!cap.isOpened()){
        spdlog::error("Error: Could not open camera with ID {}", m_config.id);
        return;
    }

    cv::Mat frame;
    while (m_isRunning.load()){ 
        if (!ConfigManager::getInstance().isWorkTime()){ 
            std::this_thread::sleep_for(std::chrono::seconds(60));
            continue;
        }
        if (!cap.read(frame) || frame.empty()){
            // Переподключение, если не может получить кадр
            spdlog::error("Camera ID {} connection lost", m_config.id);
            for (int i = 0; i < 50 && m_isRunning.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if(m_isRunning.load()) {
                cap.open(m_config.videoUrl);
            }
            continue;
        }
        
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            m_latestFrame = frame.clone();
        }
        processFrame(frame);

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    spdlog::info("Stopping processor for camera  ID: {}", m_config.id);
}


void CameraProcessor::processFrame(cv::Mat& frame) {
    cv::Rect roiRect(m_config.roi[0], m_config.roi[1], m_config.roi[2], m_config.roi[3]);
    cv::rectangle(frame, roiRect, cv::Scalar(255, 255, 0), 2);
    cv::Mat roiFrame = frame(roiRect);

    auto detections = m_humanDetector->detect(roiFrame);
    bool humanFound = !detections.empty();
    bool gestureConfirmedThisFrame = false;

    if (humanFound) {
        for (auto humanRect : detections) {
            humanRect &= cv::Rect(0, 0, roiFrame.cols, roiFrame.rows);
            if (humanRect.width <= 0 || humanRect.height <= 0) continue;
            
            cv::Rect absoluteRect = humanRect + cv::Point(roiRect.x, roiRect.y);
            cv::rectangle(frame, absoluteRect, cv::Scalar(0, 255, 0), 2);
            
            cv::Mat personFrame = roiFrame(humanRect);
            
            GestureType currentGesture = m_gestureRecognizer->recognize(personFrame);
            
            if (currentGesture == m_lastDetectedGesture && currentGesture != GestureType::NONE) {
                m_gestureCounter++;
            } else {
                m_gestureCounter = 1;
                m_lastDetectedGesture = currentGesture;
            }

            if (m_gestureCounter >= GESTURE_CONFIRMATION_FRAMES) {
                spdlog::info("Camera ID {} | Gesture {} CONFIRMED.", m_config.id, static_cast<int>(currentGesture));
                handleGesture(currentGesture);
                m_gestureCounter = 0;
                m_lastDetectedGesture = GestureType::NONE;
                gestureConfirmedThisFrame = true;
                break;
            }
        }
    } else {
        m_gestureCounter = 0;
        m_lastDetectedGesture = GestureType::NONE;
    }

    if (humanFound && !gestureConfirmedThisFrame && m_systemState->getMode() == SystemMode::AUTO) {
        auto now = std::chrono::steady_clock::now();
        if (now > m_lastRequestTime + m_cooldownDuration) {
            spdlog::info("Camera ID {} | Human detected, no gesture. Sending light ON.", m_config.id);
            m_httpClient->sendGetRequest(m_config.APIUrl); 
            m_lastRequestTime = now;
        }
    }
}

void CameraProcessor::handleGesture(GestureType gesture) {
    spdlog::info("Camera ID {} | Detected gesture {}", m_config.id, static_cast<int>(gesture));
    std::string url;

    switch (gesture)
    {
    case GestureType::ARMS_CROSSED:
        m_systemState->setMode(SystemMode::MANUAL);
        url = ConfigManager::getInstance().getGestureUrl("system_off");
        break;
    
    case GestureType::ONE_ARM_UP:
        m_systemState->setMode(SystemMode::AUTO);
        url = ConfigManager::getInstance().getGestureUrl("system_on");
        break;

    case GestureType::PEACE:
        url = ConfigManager::getInstance().getGestureUrl("peace");
        break;

    case GestureType::THUMBS_UP:
        url = ConfigManager::getInstance().getGestureUrl("thumbs_up");
        break;

    case GestureType::THUMBS_DOWN:
        url = ConfigManager::getInstance().getGestureUrl("thumbs_down");
        break;

    default:
        return;
    }

    if (!url.empty()){
        spdlog::info("Sending gesture action request to {}", url);
        m_httpClient->sendGetRequest(url);
        m_lastRequestTime = std::chrono::steady_clock::now();
    }
}
