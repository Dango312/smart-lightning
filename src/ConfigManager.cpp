// ConfigManager.cpp

#include "ConfigManager.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <regex>

ConfigManager& ConfigManager::getInstance(){
    static ConfigManager instance;
    return instance;
}

std::chrono::minutes ConfigManager::parseTime(const std::string& timeStr) const{
    std::regex timeRegex("^([0-1]?[0-9]|2[0-3]):([0-5][0-9])$");
    std::smatch match;

    if (std::regex_match(timeStr, match, timeRegex)) {
        int hours = std::stoi(match[1].str());
        int minutes = std::stoi(match[2].str());
        return std::chrono::hours(hours) + std::chrono::minutes(minutes);
    }
    throw std::runtime_error("Invalid time format " + timeStr);
}

void ConfigManager::load(const std::string& filepath){
    std::ifstream file(filepath);
    if (!file.is_open()){
        throw std::runtime_error("ConfigManage: Failed to open config file " + filepath);
    }

    nlohmann::json data;
    file >> data;

    try {
        m_cameraConfigs.clear();
        for (const auto& camJson : data.at("cameras")){
            CameraConfig config;
            config.id = camJson.at("id").get<int>();
            config.videoUrl = camJson.at("video_url").get<std::string>();
            config.APIUrl = camJson.at("APIUrl").get<std::string>();
            config.roi = camJson.at("roi").get<std::vector<int>>();
            
            m_cameraConfigs.push_back(config);
        } 

        m_gestureActions.clear();
        const auto& gesturesJson = data.at("gesture_actions");
        for (auto it = gesturesJson.begin(); it != gesturesJson.end(); ++it){
            m_gestureActions[it.key()] = it.value().get<std::string>();
        }

        const auto& nightModeJson = data.at("working_hours");
        m_workingTime.start = parseTime(nightModeJson.at("start_time").get<std::string>());
        m_workingTime.end = parseTime(nightModeJson.at("end_time").get<std::string>());
    }
    catch  (const std::exception& e){
        throw std::runtime_error("ConfigManager: Error processing config data: " + std::string(e.what()));
    }
}

bool ConfigManager::isWorkTime() const{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    const std::tm* localTime = std::localtime(&t_c);

    const auto currentMinutes = std::chrono::hours(localTime->tm_hour) + std::chrono::minutes(localTime->tm_min);

    const auto start = m_workingTime.start;
    const auto end = m_workingTime.end;

    if (start <= end){
        // Дневной диапазон
        return currentMinutes >= start && currentMinutes < end;
    }
    else {
        // Ночной диапазон
        return currentMinutes >= start || currentMinutes < end;
    }
}

const std::vector<CameraConfig>& ConfigManager::getCameraConfigs() const{
    return m_cameraConfigs;
}

std::string ConfigManager::getGestureUrl(const std::string& gestureName) const{
    auto it = m_gestureActions.find(gestureName);
    if (it != m_gestureActions.end()){
        return it->second;
    }
    return "";
}