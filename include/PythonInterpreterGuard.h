// PythonInterpreterGuard.h
#pragma once
#include <pybind11/embed.h>

namespace py = pybind11;

class PythonInterpreterGuard {
public:
    PythonInterpreterGuard() {
        py::initialize_interpreter(false); 
    }

    ~PythonInterpreterGuard() {
        py::finalize_interpreter(); 
    }

    PythonInterpreterGuard(const PythonInterpreterGuard&) = delete;
    PythonInterpreterGuard& operator=(const PythonInterpreterGuard&) = delete;
};
