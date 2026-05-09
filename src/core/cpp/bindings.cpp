// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "environment.hpp"
#include "vector_environment.hpp"

namespace py = pybind11;

PYBIND11_MODULE(maze_core, m) {
    py::class_<MazeCore>(m, "MazeCore")
        .def(py::init<int,int,int,float,float,float,float,
                      const std::string&,float,int,int,int,int,float>(),
             py::arg("grid_size"), py::arg("max_steps"), py::arg("n_food_sources"),
             py::arg("food_energy"), py::arg("initial_energy"), py::arg("energy_decay"),
             py::arg("energy_per_step"), py::arg("task_class"), py::arg("complexity_level"),
             py::arg("n_doors"), py::arg("door_open_duration"), py::arg("door_close_duration"),
             py::arg("n_buttons_per_door"), py::arg("button_break_probability"))
        .def("reset", &MazeCore::reset, py::arg("seed") = py::none())
        .def("step", &MazeCore::step)
        .def("soft_reset", &MazeCore::soft_reset);

    py::class_<VectorizedMazeEnv>(m, "VectorizedMazeEnv")
        .def(py::init<int,int,int,int,float,float,float,float,
                    const std::string&,float,int,int,int,int,float,int>(),
            py::arg("num_envs"), py::arg("grid_size"), py::arg("max_steps"),
            py::arg("n_food_sources"), py::arg("food_energy"), py::arg("initial_energy"),
            py::arg("energy_decay"), py::arg("energy_per_step"), py::arg("task_class"),
            py::arg("complexity_level"), py::arg("n_doors"), py::arg("door_open_duration"),
            py::arg("door_close_duration"), py::arg("n_buttons_per_door"),
            py::arg("button_break_probability"), py::arg("base_seed"))
        .def("reset", &VectorizedMazeEnv::reset, py::arg("seed_override") = py::none())
        .def("step", &VectorizedMazeEnv::step)
        .def("soft_reset", &VectorizedMazeEnv::soft_reset);
}
