// ConfigManager.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>


// Хранение настроек камеры
struct CameraConfig{
    int id;
    std::string videoUrl; // Путь к камере
    std::string APIUrl; // Адрес запроса
    std::vector<int> roi; // Область распознавания
};

// Хранение времени начала и конца работы в минутах от начала суток
struct WorkingTime{
    std::chrono::minutes start{0};
    std::chrono::minutes end{0};
};

class ConfigManager {
    public:
        ConfigManager(const ConfigManager&) = delete;

        static ConfigManager& getInstance();
        void operator=(const ConfigManager&) = delete;

        void load(const std::string& filepath);
        
        const std::string& getDevice() const;
        const std::vector<CameraConfig>& getCameraConfigs() const;

        bool isWorkTime() const;

        std::string getGestureUrl(const std::string& gestureName) const;

    private:
        ConfigManager() = default;
        std::string m_device;
        // Парсинг времени из строки "ЧЧ:MM"
        std::chrono::minutes parseTime(const std::string& timeStr) const;

        std::vector<CameraConfig> m_cameraConfigs;
        WorkingTime m_workingTime;
        std::map<std::string, std::string> m_gestureActions;
};
