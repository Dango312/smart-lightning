// main.cpp

#include "HttpClient.h"
#include "CameraProcessor.h"
#include "SystemState.h"
#include "ConfigManager.h"
#include "HumanDetector.h"
#include "GestureRecognizer.h"
#include "spdlog/spdlog.h"

#include <opencv2/highgui.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <memory>

int main(int argc, char** argv) {
    spdlog::info("--- Smart Lightning System Starting ---");
    try {
        ConfigManager::getInstance().load("/home/dango/smart-lightning/config.json");
        spdlog::info("Configuration loaded successfully");
        auto systemState = std::make_shared<SystemState>(); 
        auto httpClient = std::make_shared<HttpClient>();
        auto humanDetector = std::make_shared<HumanDetector>();
        auto gestureRecognizer = std::make_shared<GestureRecognizer>();

        humanDetector->loadModel("/home/dango/smart-lightning/models/human_recognizer.onnx"); // Загруза yolo
        gestureRecognizer->loadModel("models/gesture_recognizer.onnx"); // (ЗАГЛУШКА) Загрузка определителя жестов 

        std::vector<std::thread> cameraThreads;
        std::vector<std::unique_ptr<CameraProcessor>> cameraProcessors;

        const auto& cameraConfigs = ConfigManager::getInstance().getCameraConfigs();
        spdlog::info("Found {} cameras", cameraConfigs.size());

        for (const auto& camConfig : cameraConfigs) {
            auto processor = std::make_unique<CameraProcessor>(
                camConfig,
                systemState,
                httpClient,
                humanDetector,
                gestureRecognizer
            );
            cameraThreads.emplace_back(&CameraProcessor::run, processor.get());
            cameraProcessors.push_back(std::move(processor));
        }

        std::cout << "\n--- System is running ---\n";

        while(true){
            for(const auto& processor : cameraProcessors){
                cv::Mat frame = processor->getLatestFrame();
                if (!frame.empty()){
                    std::string windowName = "Camera ID " + std::to_string(processor->getConfig().id);
                    cv::imshow(windowName, frame);
                }
            }

            int key = cv::waitKey(33);
            if (key == 'q' || key == 27){
                break;
            }
        }

        for (const auto& processor: cameraProcessors){
            processor->stop();
        }
        for (auto& t : cameraThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        cv::destroyAllWindows();
    }
    catch (const std::runtime_error& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
    spdlog::info("Shutting down");
    return 0;
}