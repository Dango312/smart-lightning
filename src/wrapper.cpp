// src/wrapper.cpp
#include "wrapper.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/gil.h>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;


class PythonGestureRecognizerImpl {
public:
    PythonGestureRecognizerImpl() {
        py::gil_scoped_acquire acquire;
        try {
            py::exec("import sys; sys.path.append('..')");
            py::module_ module = py::module_::import("hand_gesture_server");
            recognize_func = module.attr("recognize_gestures");
        } catch (py::error_already_set &e) {
            throw std::runtime_error(std::string("Failed to import Python module: ") + e.what());
        }
    }

    std::string recognize(const std::vector<unsigned char>& image_bytes) {
        py::gil_scoped_acquire acquire; 
        try {
            py::bytes bytes_obj(reinterpret_cast<const char*>(image_bytes.data()), image_bytes.size());
            py::object result = recognize_func(bytes_obj);
            return result.cast<std::string>();
        } catch (py::error_already_set &e) {
            std::cerr << "Python execution error: " << e.what() << std::endl;
            return "NONE";
        }
    }

private:
    py::object recognize_func;
};

PythonGestureRecognizer::PythonGestureRecognizer() 
    : pimpl(std::make_unique<PythonGestureRecognizerImpl>()) {}

PythonGestureRecognizer::~PythonGestureRecognizer() = default;

std::string PythonGestureRecognizer::recognize(const std::vector<unsigned char>& image_bytes) {
    return pimpl->recognize(image_bytes);
}