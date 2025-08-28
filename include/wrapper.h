// include/wrapper.h
#pragma once
#include <string>
#include <vector>
#include <memory>

class PythonGestureRecognizerImpl; 

class PythonGestureRecognizer {
public:
    PythonGestureRecognizer();
    ~PythonGestureRecognizer();

    PythonGestureRecognizer(const PythonGestureRecognizer&) = delete;
    PythonGestureRecognizer& operator=(const PythonGestureRecognizer&) = delete;

    std::string recognize(const std::vector<unsigned char>& image_bytes);

private:
    std::unique_ptr<PythonGestureRecognizerImpl> pimpl;
};