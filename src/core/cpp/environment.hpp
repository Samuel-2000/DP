// environment.hpp
#pragma once
#include <vector>
#include <random>
#include <string>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <optional>
#include <map>

// Forward declaration for pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Matches Python constants
enum class TileType : uint8_t {
    EMPTY = 0,
    OBSTACLE = 1,
    FOOD_SOURCE = 2,
    FOOD = 3,
    AGENT = 4,
    DOOR_CLOSED = 5,
    DOOR_OPEN = 6,
    BUTTON = 7,
    BUTTON_BROKEN = 8
};

enum class Action : int {
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3,
    STAY = 4,
    BUTTON = 5
};

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1,T2>& p) const {
        return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
    }
};

class GridMazeWorld {
public:
    GridMazeWorld(int grid_size, int max_steps, int n_food_sources, float food_energy,
             float initial_energy, float energy_decay, float energy_per_step,
             const std::string& task_class, float complexity_level,
             int n_doors, int door_open_duration, int door_close_duration,
             int n_buttons_per_door, float button_break_probability);

    // Public interface
    std::tuple<std::vector<int>, std::map<std::string, double>> reset(std::optional<int> seed);
    std::tuple<std::vector<int>, double, bool, bool, std::map<std::string, double>> step(int action);
    std::tuple<std::vector<int>, std::map<std::string, double>> soft_reset();

    // Rendering
    py::array_t<uint8_t> render(int render_size = 512);

    // Public getters for Python
    int get_max_steps() const { return max_steps_; }
    float get_energy() const { return energy_; }
    std::string get_task_class() const { return task_class_; }
    float get_complexity_level() const { return complexity_level_; }
    int get_grid_size() const { return grid_size_; }
    int get_agent_y() const { return agent_y_; }
    int get_agent_x() const { return agent_x_; }
    int get_steps() const { return steps_; }
    const std::vector<std::vector<uint8_t>>& get_static_grid() const { return static_grid_; }
    const std::vector<std::vector<uint8_t>>& get_door_open() const { return door_open_; }
    const std::vector<std::vector<uint8_t>>& get_button_broken() const { return button_broken_; }
    const std::vector<std::pair<int,int>>& get_food_coords() const { return _food_coords; }
    const std::vector<bool>& get_food_exists() const;   // to be implemented in .cpp

private:
    // Parameters
    int grid_size_, max_steps_, n_food_sources_;
    float food_energy_, initial_energy_, energy_decay_, energy_per_step_;
    std::string task_class_;
    float complexity_level_;
    int n_doors_, door_open_duration_, door_close_duration_;
    int n_buttons_per_door_;
    float button_break_probability_;
    int n_obstacles_;

    // Grid state
    std::vector<std::vector<uint8_t>> grid_;
    std::vector<std::vector<uint8_t>> static_grid_;
    std::vector<std::vector<int8_t>> food_cache_;
    std::vector<std::vector<uint8_t>> door_open_;
    std::vector<std::vector<uint8_t>> button_broken_;
    std::vector<std::vector<uint8_t>> passable_mask_;

    struct FoodSource { int y, x, delay, exists, count; };
    std::vector<FoodSource> food_sources_;

    // Agent state
    int agent_y_, agent_x_;
    float energy_;
    int steps_;
    bool done_;
    int last_action_;

    // Doors and buttons
    struct Door {
        int y, x, open_duration, close_duration, number;
        bool requires_button, can_be_opened, is_open;
        int timer;
        bool open();
    };
    struct Button {
        int y, x, door_idx, number;
        float break_probability;
        bool is_broken;
    };
    std::vector<Door> doors_;
    std::vector<Button> buttons_;

    // Pre‑allocated buffers
    std::vector<int> _regen_buffer;
    std::vector<std::pair<int,int>> _spawn_cells;
    std::vector<std::pair<int,int>> _door_coords;
    std::vector<std::pair<int,int>> _button_coords;

    std::mt19937 rng_;

    // Helper methods
    void adjustParameters();
    void placeObstaclesWithConnectivity();
    bool isConnected(const std::unordered_set<std::pair<int,int>, PairHash>& empty) const;
    void initFoodSources();
    void initDoorsAndButtons();
    void updateDoorStates();
    void updatePassableCache();
    void rebuildPassableMask();
    bool pressButton(int by, int bx);
    std::vector<int> getObservation();
    void cacheResetState();
    bool canMoveTo(int y, int x) const;
    int manhattanDistance(int ay, int ax, int by, int bx) const;
    bool canPlaceDoorWithButtons(int y, int x, std::vector<std::pair<int,int>>& btns);
    std::vector<std::pair<int,int>> findDoorCandidates();

    void draw_cell(py::array_t<uint8_t>& img, int y, int x, int cell_size, TileType tile, bool is_door_open, bool is_button_broken);

    // Template matching
    int computeNeighborhoodMask(int y, int x) const;
    bool matchesTemplate(int y, int x, int mask) const;
    static const std::vector<std::vector<int8_t>> TEMPLATES;
    static const int8_t CENTER_IDX = 4;
};