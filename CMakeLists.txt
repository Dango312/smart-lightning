cmake_minimum_required(VERSION 3.10)
project(SmartLightning CXX)
set(CMAKE_CXX_STANDART 17)
set(CMAKE_CXX_STANDART_REQUIRED ON)

find_package(cpr REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)

set(ONNXRUNTIME_DIR "/home/dango/libs/onnxruntime-linux-x64-1.22.0") # Здесь надо указать свой путь к onnx
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          PATH_SUFFIXES include
          PATHS ${ONNXRUNTIME_DIR})
find_library(ONNXRUNTIME_LIBRARY
             NAMES onnxruntime
             PATH_SUFFIXES lib
             PATHS ${ONNXRUNTIME_DIR})


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})


add_executable(
    smart_lighting
    src/main.cpp
    src/CameraProcessor.cpp
    src/ConfigManager.cpp
    src/HttpClient.cpp
    src/GestureRecognizer.cpp
    src/HumanDetector.cpp
)

target_link_libraries(
    smart_lighting
    ${OpenCV_LIBS}
    cpr::cpr
    Threads::Threads
    ${ONNXRUNTIME_LIBRARY}
    spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>
)

message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")