#pragma once
#include <vector>
#include <memory>
#include <optional>
#include "environment.hpp"

class VectorizedMazeEnv {
public:
    VectorizedMazeEnv(int num_envs,
                   int grid_size, int max_steps, int n_food_sources, float food_energy,
                   float initial_energy, float energy_decay, float energy_per_step,
                   const std::string& task_class, float complexity_level,
                   int n_doors, int door_open_duration, int door_close_duration,
                   int n_buttons_per_door, float button_break_probability,
                   int base_seed);

    std::tuple<std::vector<std::vector<int>>, std::vector<std::map<std::string, double>>> reset(std::optional<int> seed_override = std::nullopt);
    std::tuple<std::vector<std::vector<int>>, std::vector<float>, std::vector<bool>, std::vector<bool>, std::vector<std::map<std::string, double>>> step(const std::vector<int>& actions);
    std::tuple<std::vector<std::vector<int>>, std::vector<std::map<std::string, double>>> soft_reset();

private:
    int num_envs_;
    int base_seed_;
    int reset_counter_;
    std::vector<std::unique_ptr<MazeCore>> envs_;
};