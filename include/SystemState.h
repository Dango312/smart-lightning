// SystemState.h
#pragma once

#include <atomic>

enum class SystemMode{
    AUTO,
    MANUAL
};

class SystemState{
    public:
        SystemState() : m_currentMode(SystemMode::AUTO) {}

        void setMode(SystemMode mode){
            m_currentMode.store(mode);
        }

        SystemMode getMode() const{
            return m_currentMode.load();
        }
    private:
        std::atomic<SystemMode> m_currentMode;
};