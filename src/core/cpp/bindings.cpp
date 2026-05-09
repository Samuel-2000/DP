#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "environment.hpp"
#include "vector_environment.hpp"

namespace py = pybind11;

PYBIND11_MODULE(maze_core, m) {
    // ------------------------------------------------------------------
    // StepInfo – exposed as a Python object with read-only attributes
    // ------------------------------------------------------------------
    py::class_<StepInfo>(m, "StepInfo")
        .def_readonly("energy", &StepInfo::energy)
        .def_readonly("steps", &StepInfo::steps)
        .def_readonly("food_collected", &StepInfo::food_collected)
        .def_readonly("button_pressed", &StepInfo::button_pressed)
        .def_readonly("complexity_level", &StepInfo::complexity_level)
        .def_readonly("n_doors", &StepInfo::n_doors)
        .def_readonly("n_buttons", &StepInfo::n_buttons)
        .def_readonly("n_doors_active", &StepInfo::n_doors_active)
        .def_readonly("n_buttons_working", &StepInfo::n_buttons_working)
        .def("__repr__", [](const StepInfo& info) {
            return "<StepInfo energy=" + std::to_string(info.energy) +
                   " steps=" + std::to_string(info.steps) +
                   " food=" + std::to_string(info.food_collected) + ">";
        });

    // ------------------------------------------------------------------
    // GridMazeWorld – single environment (fast path)
    // ------------------------------------------------------------------
    py::class_<GridMazeWorld>(m, "GridMazeWorld")
        .def(py::init<int,int,int,float,float,float,float,
                      const std::string&,float,int,int,int,int,float>(),
             py::arg("grid_size"), py::arg("max_steps"), py::arg("n_food_sources"),
             py::arg("food_energy"), py::arg("initial_energy"), py::arg("energy_decay"),
             py::arg("energy_per_step"), py::arg("task_class"), py::arg("complexity_level"),
             py::arg("n_doors"), py::arg("door_open_duration"), py::arg("door_close_duration"),
             py::arg("n_buttons_per_door"), py::arg("button_break_probability"))
        .def("reset", &GridMazeWorld::reset, py::arg("seed") = py::none())
        .def("step", &GridMazeWorld::step)          // returns tuple with StepInfo
        .def("soft_reset", &GridMazeWorld::soft_reset)
        .def("render", &GridMazeWorld::render, py::arg("render_size") = 512)
        .def("info_to_map", &GridMazeWorld::info_to_map)   // optional dict conversion
        .def_property_readonly("max_steps", &GridMazeWorld::get_max_steps)
        .def_property_readonly("energy", &GridMazeWorld::get_energy)
        .def_property_readonly("task_class", &GridMazeWorld::get_task_class)
        .def_property_readonly("complexity_level", &GridMazeWorld::get_complexity_level)
        .def_property_readonly("grid_size", &GridMazeWorld::get_grid_size)
        .def_property_readonly("agent_y", &GridMazeWorld::get_agent_y)
        .def_property_readonly("agent_x", &GridMazeWorld::get_agent_x)
        .def_property_readonly("steps", &GridMazeWorld::get_steps)
        .def_property_readonly("static_grid", &GridMazeWorld::get_static_grid)
        .def_property_readonly("door_open", &GridMazeWorld::get_door_open)
        .def_property_readonly("button_broken", &GridMazeWorld::get_button_broken)
        .def_property_readonly("food_coords", &GridMazeWorld::get_food_coords)
        .def_property_readonly("food_exists", &GridMazeWorld::get_food_exists);

    // ------------------------------------------------------------------
    // VectorizedMazeEnv – parallel environments (still returns StepInfo)
    // ------------------------------------------------------------------
    py::class_<VectorizedMazeEnv>(m, "VectorizedMazeEnv")
        .def(py::init<int,int,int,int,float,float,float,float,
                      const std::string&,float,int,int,int,int,float,int>(),
            py::arg("num_envs"), py::arg("grid_size"), py::arg("max_steps"),
            py::arg("n_food_sources"), py::arg("food_energy"), py::arg("initial_energy"),
            py::arg("energy_decay"), py::arg("energy_per_step"), py::arg("task_class"),
            py::arg("complexity_level"), py::arg("n_doors"), py::arg("door_open_duration"),
            py::arg("door_close_duration"), py::arg("n_buttons_per_door"),
            py::arg("button_break_probability"), py::arg("base_seed"))
        .def("__len__", &VectorizedMazeEnv::size)
        .def("__getitem__", [](VectorizedMazeEnv &self, size_t i) -> GridMazeWorld& {
            if (i >= self.size()) throw py::index_error();
            return *self[i];
        }, py::return_value_policy::reference_internal)
        .def("reset", &VectorizedMazeEnv::reset, py::arg("seed_override") = py::none())
        .def("step", &VectorizedMazeEnv::step)      // returns vector<StepInfo>
        .def("soft_reset", &VectorizedMazeEnv::soft_reset)
        .def("soft_reset_all", &VectorizedMazeEnv::soft_reset)   // alias for trainer
        .def("close", &VectorizedMazeEnv::close);
}