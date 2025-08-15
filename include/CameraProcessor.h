// CameraProcessor.h

#pragma once

#include "ConfigManager.h"
#include "SystemState.h"
#include "HttpClient.h"
#include "HumanDetector.h"
#include "GestureRecognizer.h"

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>

class CameraProcessor{
    public:
        CameraProcessor(
            const CameraConfig& config,
            std::shared_ptr<SystemState> systemState,
            std::shared_ptr<HttpClient> httpClient,
            std::shared_ptr<HumanDetector> humanDetector,
            std::shared_ptr<GestureRecognizer> gestureRecognizer
        );

        void run();

        void stop();
        const CameraConfig& getConfig() const { return m_config; }
        cv::Mat getLatestFrame();

    private:
        void processFrame(cv::Mat& frame);
        void handleGesture(GestureType gesture);

        CameraConfig m_config;
        std::shared_ptr<SystemState> m_systemState;
        std::shared_ptr<HttpClient> m_httpClient;
        std::shared_ptr<HumanDetector> m_humanDetector;
        std::shared_ptr<GestureRecognizer> m_gestureRecognizer;
        
        std::atomic<bool> m_isRunning;

        // Контроль частоты запросов
        std::chrono::steady_clock::time_point m_lastRequestTime;
        std::chrono::seconds m_cooldownDuration;

        std::mutex m_frameMutex;
        cv::Mat m_latestFrame;
};

