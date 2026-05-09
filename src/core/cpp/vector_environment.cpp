// vector_environment.cpp
#include "vector_environment.hpp"
#include <algorithm>

VectorizedMazeEnv::VectorizedMazeEnv(int num_envs,
                               int grid_size, int max_steps, int n_food_sources, float food_energy,
                               float initial_energy, float energy_decay, float energy_per_step,
                               const std::string& task_class, float complexity_level,
                               int n_doors, int door_open_duration, int door_close_duration,
                               int n_buttons_per_door, float button_break_probability,
                               int base_seed)
    : num_envs_(num_envs), base_seed_(base_seed), reset_counter_(0) {
    for (int i = 0; i < num_envs_; ++i) {
        envs_.emplace_back(std::make_unique<GridMazeWorld>(
            grid_size, max_steps, n_food_sources, food_energy,
            initial_energy, energy_decay, energy_per_step,
            task_class, complexity_level,
            n_doors, door_open_duration, door_close_duration,
            n_buttons_per_door, button_break_probability));
    }
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::map<std::string, double>>>
VectorizedMazeEnv::reset(std::optional<int> seed_override) {
    std::vector<std::vector<int>> all_obs(num_envs_);
    std::vector<std::map<std::string, double>> all_infos(num_envs_);
    for (int i = 0; i < num_envs_; ++i) {
        int seed = seed_override.value_or(base_seed_ + reset_counter_ * num_envs_ + i);
        auto [obs, info] = envs_[i]->reset(seed);
        all_obs[i] = obs;
        all_infos[i] = info;
    }
    ++reset_counter_;
    return {all_obs, all_infos};
}

std::tuple<std::vector<std::vector<int>>, std::vector<float>, std::vector<bool>, std::vector<bool>, std::vector<std::map<std::string, double>>>
VectorizedMazeEnv::step(const std::vector<int>& actions) {
    std::vector<std::vector<int>> all_obs(num_envs_);
    std::vector<float> all_rewards(num_envs_);
    std::vector<bool> all_terminated(num_envs_);
    std::vector<bool> all_truncated(num_envs_);
    std::vector<std::map<std::string, double>> all_infos(num_envs_);
    for (int i = 0; i < num_envs_; ++i) {
        auto [obs, reward, terminated, truncated, info] = envs_[i]->step(actions[i]);
        all_obs[i] = obs;
        all_rewards[i] = reward;
        all_terminated[i] = terminated;
        all_truncated[i] = truncated;
        all_infos[i] = info;
    }
    return {all_obs, all_rewards, all_terminated, all_truncated, all_infos};
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::map<std::string, double>>>
VectorizedMazeEnv::soft_reset() {
    std::vector<std::vector<int>> all_obs(num_envs_);
    std::vector<std::map<std::string, double>> all_infos(num_envs_);
    for (int i = 0; i < num_envs_; ++i) {
        auto [obs, info] = envs_[i]->soft_reset();
        all_obs[i] = obs;
        all_infos[i] = info;
    }
    return {all_obs, all_infos};
}