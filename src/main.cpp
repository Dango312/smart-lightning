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
#include <filesystem>
#include <pybind11/embed.h>
#include "PythonInterpreterGuard.h"
#include <csignal>
#include <atomic>

namespace py = pybind11;

std::atomic<bool> g_shutdown_flag(false);
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        if (g_shutdown_flag.load()) {
            return;
        }
        spdlog::info("\nShutdown signal ({}) received. Initiating graceful shutdown...", signal);
        g_shutdown_flag.store(true);
    }
}

int main(int argc, char** argv) {
    
    PythonInterpreterGuard guard;
    //py::gil_scoped_release main_gil_release; 
    std::filesystem::path executable_path(argv[0]);
    std::filesystem::path project_root = executable_path.parent_path().parent_path();
    
    std::filesystem::path config_path = project_root / "config.json";
    //signal(SIGINT, signal_handler);
    //signal(SIGTERM, signal_handler);
    spdlog::info("--- Smart Lightning System Starting ---");
    try {
        ConfigManager::getInstance().load(config_path.string());
        spdlog::info("Configuration loaded successfully");
        auto systemState = std::make_shared<SystemState>(); 
        auto httpClient = std::make_shared<HttpClient>();
        auto humanDetector = std::make_shared<HumanDetector>();
        auto gestureRecognizer = std::make_shared<GestureRecognizer>();

        // Загруза моделей
        humanDetector->loadModel((project_root / "models/human_recognizer.onnx").string()); // Загруза yolo

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
        // Отображение окон с камерами (закомментировано для работы на сервере)
        
        {
            py::gil_scoped_release release_gil;
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
        }
        

        // Для сервера (раскомментировать)
        //while(!g_shutdown_flag.load()) {
        //    std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //}
        // Завершение работы
        for (const auto& processor: cameraProcessors){
            processor->stop();
        }
        {
            py::gil_scoped_release release_gil_on_join;
            for (auto& t : cameraThreads) {
                if (t.joinable()) {
                    t.join();
                }
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