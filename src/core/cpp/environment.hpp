#pragma once
#include <vector>
#include <random>
#include <string>
#include <queue>
#include <tuple>
#include <optional>
#include <map>
#include <array>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// ------------------------------------------------------------------
// Aligned allocator for SIMD (32‑byte alignment)
// ------------------------------------------------------------------
template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        void* ptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();
#endif
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

struct StepInfo {
    double energy;
    int steps;
    int food_collected;
    int button_pressed;
    double complexity_level;
    int n_doors;
    int n_buttons;
    int n_doors_active;
    int n_buttons_working;
};

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
    LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3, STAY = 4, BUTTON = 5
};

class GridMazeWorld {
public:
    GridMazeWorld(int grid_size, int max_steps, int n_food_sources, float food_energy,
                  float initial_energy, float energy_decay, float energy_per_step,
                  const std::string& task_class, float complexity_level,
                  int n_doors, int door_open_duration, int door_close_duration,
                  int n_buttons_per_door, float button_break_probability);

    std::tuple<std::vector<int>, std::map<std::string, double>> reset(std::optional<int> seed);
    std::tuple<std::vector<int>, std::map<std::string, double>> soft_reset();
    std::tuple<const std::vector<int>&, double, bool, bool, StepInfo> step(int action);
    const StepInfo& get_last_info() const { return last_info_; }
    py::array_t<uint8_t> render(int render_size = 512);

    int get_max_steps() const { return max_steps_; }
    float get_energy() const { return energy_; }
    std::string get_task_class() const { return task_class_; }
    float get_complexity_level() const { return complexity_level_; }
    int get_grid_size() const { return grid_size_; }
    int get_agent_y() const { return agent_y_; }
    int get_agent_x() const { return agent_x_; }
    int get_steps() const { return steps_; }
    const std::vector<uint8_t, AlignedAllocator<uint8_t>>& get_static_grid() const { return static_grid_; }
    const std::vector<uint8_t, AlignedAllocator<uint8_t>>& get_door_open() const { return door_open_; }
    const std::vector<uint8_t, AlignedAllocator<uint8_t>>& get_button_broken() const { return button_broken_; }
    const std::vector<std::pair<int,int>>& get_food_coords() const { return _food_coords; }
    const std::vector<bool>& get_food_exists() const { return food_exists_cache_; }

private:
    int grid_size_, max_steps_, n_food_sources_;
    float food_energy_, initial_energy_, energy_decay_, energy_per_step_;
    std::string task_class_;
    float complexity_level_;
    int n_doors_, door_open_duration_, door_close_duration_;
    int n_buttons_per_door_;
    float button_break_probability_;
    int n_obstacles_;
    int total_cells_;

    // Aligned vectors for SIMD
    std::vector<uint8_t, AlignedAllocator<uint8_t>> grid_;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> static_grid_;
    std::vector<int8_t, AlignedAllocator<int8_t>> food_cache_;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> door_open_;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> button_broken_;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> passable_mask_;

    struct FoodSource { 
        int y, x, delay, exists, count;
        int regrow_step;
    };
    std::vector<FoodSource> food_sources_;
    std::vector<bool> food_exists_cache_;

    int agent_y_, agent_x_;
    float energy_;
    int steps_;
    bool done_;
    int last_action_;

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
    int n_doors_active_, n_buttons_working_;

    // Buffers (no alignment required for these temporary arrays)
    std::vector<int> bfs_queue_;
    std::vector<int> bfs_dist_;
    std::vector<int> labels_;
    std::vector<std::pair<int,int>> empty_cells_;
    std::vector<std::pair<int,int>> spawn_cells_;
    std::vector<uint8_t> pass_buf_;
    std::vector<int> dist_buf_;
    std::vector<int> queue_buf_;
    std::vector<int> labels_buf_;
    std::vector<int> stack_buf_;

    std::vector<int> food_cell_to_idx_;
    std::vector<int> active_pos_;
    std::vector<int> active_depleted_food_;
    std::priority_queue<std::pair<int,int>,
                        std::vector<std::pair<int,int>>,
                        std::greater<std::pair<int,int>>> regrow_heap_;
    int step_counter_;

    std::vector<int> button_at_cell_;
    std::vector<int> door_at_cell_;
    std::vector<int> working_buttons_per_door_;

    mutable std::vector<int> obs_buffer_;
    StepInfo last_info_;

    std::vector<std::pair<int,int>> _food_coords;
    std::vector<std::pair<int,int>> _door_coords;
    std::vector<std::pair<int,int>> _button_coords;

    // Articulation analysis
    std::vector<bool> is_articulation_final_;
    std::vector<int> artic_comp_count_;
    std::vector<int> subtree_empty_;
    std::vector<std::vector<int>> artic_comp_sizes_;
    int total_empty_;

    int nextDoorNumber_;
    std::mt19937 rng_;

    // Pre‑computed neighbour offsets (for observation)
    static constexpr int dy8_[8] = {-1,-1,-1,0,0,1,1,1};
    static constexpr int dx8_[8] = {-1,0,1,-1,1,-1,0,1};

    inline int idx(int y, int x) const { return y * grid_size_ + x; }

    void placeObstaclesWithConnectivity();
    void initFoodSources();
    void initDoorsAndButtons();
    void initPassableCache();
    void updateDoorStates();
    inline void setDoorOpen(int doorIdx, bool open);
    bool canMoveTo(int y, int x) const { return passable_mask_[idx(y,x)] == 1; }
    int manhattanDistance(int y1, int x1, int y2, int x2) const;
    bool pressButton(int by, int bx);
    const std::vector<int>& getObservation();

    void labelConnectedComponents(const std::vector<uint8_t>& pass_mask, std::vector<int>& labels, int& nlabels);
    void bfsReachable(int sy, int sx, int maxdist, const std::vector<uint8_t>& pass_mask,
                      std::vector<int>& dist, std::vector<int>& queue);
    void cacheResetState();

    void computeFinalArticulationPoints();
    bool placeDoorWithButtons(int y, int x);
    void placeButtonsForDoor(int doorIdx);   // new: second‑phase button placement
    void connectIsolatedCells();
    bool checkComponentsMinSize(int y, int x, int minEmpty);
};