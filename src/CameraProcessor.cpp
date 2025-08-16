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
        m_isRunning(true) {

    
    int cooldownSec=5;
    m_gamma = 0.8;

    m_cooldownDuration = std::chrono::seconds(cooldownSec);
    m_lastRequestTime = std::chrono::steady_clock::now() - m_cooldownDuration;
    initGammaCorrection();
}

void CameraProcessor::initGammaCorrection(){
    m_gammaLut.create(1, 256, CV_8U);
    uchar* p = m_gammaLut.ptr();
    for (int i = 0; i < 256; ++i){
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, m_gamma) * 255.0);
    }
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
            std::this_thread::sleep_for(std::chrono::seconds(5)); 
            cap.open(m_config.videoUrl);
            continue;
        }
        
        {
            cv::Mat corrected_frame = preprocessing(frame);
            std::lock_guard<std::mutex> lock(m_frameMutex);
            m_latestFrame = corrected_frame.clone();
        }
        processFrame(frame);

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    spdlog::info("Stopping processor for camera  ID: {}", m_config.id);
}

cv::Mat CameraProcessor::preprocessing(cv::Mat& frame) {
    int patch_size = 15;
    float omega = 0.1f;
    float t_min = 0.1f;
    double guided_filter_eps = 1e-3; 

    cv::Mat I;
    frame.convertTo(I, CV_32FC3, 1.0 / 255.0);

    cv::Mat dark_channel;
    {
        std::vector<cv::Mat> channels;
        cv::split(I, channels);
        dark_channel = cv::min(channels[0], cv::min(channels[1], channels[2]));
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patch_size, patch_size));
        cv::erode(dark_channel, dark_channel, kernel);
    }

    cv::Scalar A;
    {
        cv::Point max_loc;
        cv::minMaxLoc(dark_channel, nullptr, nullptr, nullptr, &max_loc);
        
        A = I.at<cv::Vec3f>(max_loc);
    }
    
    cv::Mat transmission;
    {
        cv::Mat im_norm_by_A;
        cv::divide(I, A, im_norm_by_A);
        
        std::vector<cv::Mat> channels;
        cv::split(im_norm_by_A, channels);
        transmission = cv::min(channels[0], cv::min(channels[1], channels[2]));
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patch_size, patch_size));
        cv::erode(transmission, transmission, kernel);
        
        transmission = 1.0 - omega * transmission;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
    cv::ximgproc::guidedFilter(gray, transmission, transmission, patch_size, guided_filter_eps);

    cv::Mat J;
    {
        cv::Mat t;
        cv::max(transmission, t_min, t);
        
        cv::Mat t_3ch;
        cv::cvtColor(t, t_3ch, cv::COLOR_GRAY2BGR);
        
        J = (I - cv::Mat(I.size(), I.type(), A)) / t_3ch + cv::Mat(I.size(), I.type(), A);
    }

    cv::Mat output;
    J.convertTo(output, CV_8UC3, 255.0);

    cv::Mat lab;
    cv::cvtColor(output, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels(3);
    cv::split(lab, lab_channels);
    
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(3.0);
    clahe->apply(lab_channels[0], lab_channels[0]);
    
    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);

    return output;
}


void CameraProcessor::processFrame(cv::Mat& frame) {
    cv::Rect roiRect(m_config.roi[0], m_config.roi[1], m_config.roi[2], m_config.roi[3]);
    
    cv::rectangle(frame, roiRect, cv::Scalar(255, 255, 0), 2);
    cv::Mat roiFrame = frame(roiRect);

    cv::Mat corrected_frame = preprocessing(roiFrame);

    auto detections = m_humanDetector->detect(corrected_frame);
    bool humanFound = !detections.empty();
    bool gestureHandled = false;

    if (humanFound){
        for (const auto& humanRect : detections) {
            cv::Rect absoluteRect = humanRect + cv::Point(roiRect.x, roiRect.y);
            cv::rectangle(frame, absoluteRect, cv::Scalar(0, 255, 0), 2);
            
            GestureType gesture = m_gestureRecognizer->recognize(roiFrame, humanRect);
            if (gesture != GestureType::NONE) {
                handleGesture(gesture);
                gestureHandled = true;
                break;
            }
        }
    }

    if (humanFound && !gestureHandled && m_systemState->getMode() == SystemMode::AUTO){
        auto now = std::chrono::steady_clock::now();
        if (now > m_lastRequestTime + m_cooldownDuration) {
            spdlog::info("Camera ID {} | Human detected", m_config.id);
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

    case GestureType::CUSTOM_1:
        url = ConfigManager::getInstance().getGestureUrl("gesture_1");
        break;

    default:
        return;
    }

    if (!url.empty()){
        spdlog::info("Sending gesture action request to {}", url);
        m_httpClient->sendGetRequest(url);
    }
}
