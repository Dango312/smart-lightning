cmake_minimum_required(VERSION 3.10)
project(SmartLightning CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cpr REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)

find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA Toolkit found at: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(WARNING "CUDA Toolkit not found. Building with CPU support only.")
endif()

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to the extracted ONNX Runtime directory (CPU or GPU)")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_DIR}
          PATH_SUFFIXES include)
find_library(ONNXRUNTIME_LIBRARY
             NAMES onnxruntime
             HINTS ${ONNXRUNTIME_DIR}
             PATH_SUFFIXES lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found! Please specify its path with -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()


add_executable(
    smart_lightning
    src/main.cpp
    src/CameraProcessor.cpp
    src/ConfigManager.cpp
    src/HttpClient.cpp
    src/GestureRecognizer.cpp
    src/HumanDetector.cpp
)

target_link_libraries(
    smart_lightning PRIVATE
    ${OpenCV_LIBS}
    cpr::cpr
    Threads::Threads
    ${ONNXRUNTIME_LIBRARY}
    spdlog::spdlog
)

if(CUDA_FOUND)
    target_link_libraries(smart_lightning PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY})
endif()


message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")