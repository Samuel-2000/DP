#include "environment.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <iomanip>
#include <sstream>

const std::vector<std::vector<int8_t>> GridMazeWorld::TEMPLATES = {
    {-1, 0, -1, 1, 0, 1, -1, 0, -1},
    {-1, 1, -1, 0, 0, 0, -1, 1, -1},
    {-1, 0, 1, 0, 0, 0, 1, 0, -1},
    {1, 0, -1, 0, 0, 0, -1, 0, 1},
    {-1, 1, -1, 0, 0, -1, 1, 0, -1},
    {-1, -1, -1, 0, 0, 1, 1, 0, -1},
    {-1, 0, 1, 1, 0, 0, -1, -1, -1},
    {-1, 0, 1, -1, 0, 0, -1, 1, -1},
    {-1, 1, -1, -1, 0, 0, -1, 0, 1},
    {-1, -1, -1, 1, 0, 0, -1, 0, 1},
    {1, 0, -1, 0, 0, -1, -1, 1, -1},
    {1, 0, -1, 0, 0, 1, -1, -1, -1}
};
const int8_t GridMazeWorld::CENTER_IDX = 4;

bool GridMazeWorld::Door::open() {
    if (can_be_opened) {
        is_open = true;
        timer = 0;
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------
// Constructor – allocates all reusable buffers
// ----------------------------------------------------------------------
GridMazeWorld::GridMazeWorld(int gs, int ms, int nf, float fe, float ie, float ed, float eps,
                             const std::string& tc, float cl, int nd, int dod, int dcd,
                             int nbpd, float bbp)
    : grid_size_(gs), max_steps_(ms), n_food_sources_(nf), food_energy_(fe),
      initial_energy_(ie), energy_decay_(ed), energy_per_step_(eps),
      task_class_(tc), complexity_level_(cl), n_doors_(nd), door_open_duration_(dod),
      door_close_duration_(dcd), n_buttons_per_door_(nbpd), button_break_probability_(bbp),
      total_cells_(gs * gs), rng_(std::random_device{}()) {

    float obstacle_fraction = 0.15f + cl * 0.15f;
    n_obstacles_ = static_cast<int>((gs - 2) * (gs - 2) * obstacle_fraction);

    // Allocate all reusable buffers once
    bfs_queue_.resize(total_cells_);
    bfs_dist_.resize(total_cells_);
    labels_.resize(total_cells_);
    stack_buf_.resize(total_cells_);
    pass_buf_.resize(total_cells_);
    dist_buf_.resize(total_cells_);
    queue_buf_.resize(total_cells_);
    labels_buf_.resize(total_cells_);
    near_door_buf_.resize(total_cells_);

    // Food optimisation buffers – will be sized later
    if (n_food_sources_ > 0) {
        food_cell_to_idx_.assign(total_cells_, -1);
    }

    // Lookup tables for O(1) button/door access
    button_at_cell_.assign(total_cells_, -1);
    door_at_cell_.assign(total_cells_, -1);

    // Observation buffer – fixed size 10
    obs_buffer_.resize(10);
    last_info_ = StepInfo{}; // zero initialise

    // Adjust parameters based on task class
    if (task_class_ == "basic") {
        n_doors_ = 0; n_buttons_per_door_ = 0; button_break_probability_ = 0.0f;
    } else if (task_class_ == "doors") {
        if (n_doors_ <= 0) n_doors_ = std::max(1, static_cast<int>(complexity_level_ * 3));
        n_buttons_per_door_ = 0; button_break_probability_ = 0.0f;
    } else if (task_class_ == "buttons") {
        if (n_doors_ <= 0) n_doors_ = std::max(1, static_cast<int>(complexity_level_ * 3));
        if (n_buttons_per_door_ <= 0) n_buttons_per_door_ = 4;
        if (button_break_probability_ < 0) button_break_probability_ = complexity_level_ * 0.2f;
    } else if (task_class_ == "complex") {
        if (n_doors_ <= 0) n_doors_ = std::max(2, static_cast<int>(complexity_level_ * 4));
        if (n_buttons_per_door_ <= 0) n_buttons_per_door_ = 4;
        if (button_break_probability_ < 0) button_break_probability_ = complexity_level_ * 0.3f;
    }
}

// ----------------------------------------------------------------------
// Optimised obstacle placement – connected free space in O(free cells)
// ----------------------------------------------------------------------
void GridMazeWorld::placeObstaclesWithConnectivity() {
    grid_.assign(total_cells_, static_cast<uint8_t>(TileType::EMPTY));
    for (int i = 0; i < grid_size_; ++i) {
        grid_[idx(0, i)] = static_cast<uint8_t>(TileType::OBSTACLE);
        grid_[idx(grid_size_ - 1, i)] = static_cast<uint8_t>(TileType::OBSTACLE);
        grid_[idx(i, 0)] = static_cast<uint8_t>(TileType::OBSTACLE);
        grid_[idx(i, grid_size_ - 1)] = static_cast<uint8_t>(TileType::OBSTACLE);
    }

    if (n_obstacles_ <= 0) return;

    std::vector<int> empty_indices;
    empty_indices.reserve(total_cells_);
    for (int y = 1; y < grid_size_ - 1; ++y)
        for (int x = 1; x < grid_size_ - 1; ++x)
            if (grid_[idx(y, x)] == static_cast<uint8_t>(TileType::EMPTY))
                empty_indices.push_back(idx(y, x));

    auto is_connected = [&](const std::vector<int>& cells) -> bool {
        if (cells.empty()) return true;
        std::vector<uint8_t>& visited = pass_buf_;
        std::fill(visited.begin(), visited.end(), 0);
        int qhead = 0, qtail = 0;
        int start = cells[0];
        visited[start] = 1;
        queue_buf_[qtail++] = start;

        while (qhead < qtail) {
            int cur = queue_buf_[qhead++];
            int y = cur / grid_size_, x = cur % grid_size_;
            if (y > 0) {
                int nid = idx(y-1, x);
                if (grid_[nid] == static_cast<uint8_t>(TileType::EMPTY) && !visited[nid]) {
                    visited[nid] = 1;
                    queue_buf_[qtail++] = nid;
                }
            }
            if (y < grid_size_-1) {
                int nid = idx(y+1, x);
                if (grid_[nid] == static_cast<uint8_t>(TileType::EMPTY) && !visited[nid]) {
                    visited[nid] = 1;
                    queue_buf_[qtail++] = nid;
                }
            }
            if (x > 0) {
                int nid = idx(y, x-1);
                if (grid_[nid] == static_cast<uint8_t>(TileType::EMPTY) && !visited[nid]) {
                    visited[nid] = 1;
                    queue_buf_[qtail++] = nid;
                }
            }
            if (x < grid_size_-1) {
                int nid = idx(y, x+1);
                if (grid_[nid] == static_cast<uint8_t>(TileType::EMPTY) && !visited[nid]) {
                    visited[nid] = 1;
                    queue_buf_[qtail++] = nid;
                }
            }
        }
        for (int c : cells) if (!visited[c]) return false;
        return true;
    };

    int placed = 0, attempts = 0;
    const int maxAttempts = 100;
    while (placed < n_obstacles_ && attempts < maxAttempts) {
        ++attempts;
        if (empty_indices.empty()) break;
        std::uniform_int_distribution<int> dist(0, empty_indices.size() - 1);
        int pos = dist(rng_);
        int cell = empty_indices[pos];
        grid_[cell] = static_cast<uint8_t>(TileType::OBSTACLE);
        empty_indices.erase(empty_indices.begin() + pos);
        if (is_connected(empty_indices)) {
            ++placed;
        } else {
            grid_[cell] = static_cast<uint8_t>(TileType::EMPTY);
            empty_indices.push_back(cell);
        }
    }
}

// ----------------------------------------------------------------------
// Food source initialisation – also builds fast lookup maps
// ----------------------------------------------------------------------
void GridMazeWorld::initFoodSources() {
    empty_cells_.clear();
    for (int y = 1; y < grid_size_ - 1; ++y)
        for (int x = 1; x < grid_size_ - 1; ++x)
            if (grid_[idx(y, x)] == static_cast<uint8_t>(TileType::EMPTY))
                empty_cells_.emplace_back(y, x);

    if (empty_cells_.empty()) return;
    std::shuffle(empty_cells_.begin(), empty_cells_.end(), rng_);
    int n_food = std::min(static_cast<int>(empty_cells_.size()), n_food_sources_);
    food_sources_.clear();
    std::uniform_int_distribution<int> delayDist(10, 30);

    float centre = (grid_size_ - 1) * 0.5f;
    std::vector<float> dists(empty_cells_.size());
    for (size_t i = 0; i < empty_cells_.size(); ++i) {
        auto [y, x] = empty_cells_[i];
        dists[i] = std::abs(y - centre) + std::abs(x - centre);
    }
    std::vector<int> idx_vec(empty_cells_.size());
    std::iota(idx_vec.begin(), idx_vec.end(), 0);

    float p = std::max(0.05f, 1.0f - complexity_level_);
    int n_centre = static_cast<int>((1.0f - p) * n_food);
    n_centre = std::max(0, std::min(n_centre, n_food));
    if (n_centre > 0) {
        std::partial_sort(idx_vec.begin(), idx_vec.begin() + n_centre, idx_vec.end(),
            [&](int a, int b) { return dists[a] < dists[b]; });
    }

    std::vector<bool> used(empty_cells_.size(), false);
    for (int i = 0; i < n_food; ++i) {
        int chosen;
        if (i < n_centre)
            chosen = idx_vec[i];
        else {
            int k = std::max(2, static_cast<int>(std::sqrt(empty_cells_.size() / std::max(n_food, 1))));
            std::uniform_int_distribution<int> offy(0, k-1), offx(0, k-1);
            int oy = offy(rng_), ox = offx(rng_);
            std::vector<int> spreadPool;
            for (size_t j = 0; j < empty_cells_.size(); ++j) {
                if (used[j]) continue;
                auto [y, x] = empty_cells_[j];
                if ((y - oy) % k == 0 && (x - ox) % k == 0)
                    spreadPool.push_back(j);
            }
            if (spreadPool.empty()) {
                for (size_t j = 0; j < empty_cells_.size(); ++j)
                    if (!used[j]) spreadPool.push_back(j);
            }
            std::uniform_int_distribution<int> spd(0, spreadPool.size() - 1);
            chosen = spreadPool[spd(rng_)];
        }
        used[chosen] = true;
        auto [y, x] = empty_cells_[chosen];
        int delay = delayDist(rng_);
        food_sources_.push_back({y, x, delay, 1, 0});
        grid_[idx(y, x)] = static_cast<uint8_t>(TileType::FOOD_SOURCE);
    }

    food_cell_to_idx_.assign(total_cells_, -1);
    for (size_t i = 0; i < food_sources_.size(); ++i) {
        const auto& fs = food_sources_[i];
        food_cell_to_idx_[idx(fs.y, fs.x)] = static_cast<int>(i);
    }
    active_pos_.assign(food_sources_.size(), -1);
    active_depleted_food_.clear();

    food_cache_.assign(total_cells_, 0);
    for (const auto& fs : food_sources_)
        if (fs.exists)
            food_cache_[idx(fs.y, fs.x)] = 1;

    food_exists_cache_.resize(food_sources_.size());
    for (size_t i = 0; i < food_sources_.size(); ++i)
        food_exists_cache_[i] = (food_sources_[i].exists == 1);
}

// ----------------------------------------------------------------------
// Door candidate search
// ----------------------------------------------------------------------
std::vector<std::pair<int,int>> GridMazeWorld::findDoorCandidates() {
    std::vector<std::pair<int,int>> candidates;
    candidates.reserve(empty_cells_.size());
    std::fill(near_door_buf_.begin(), near_door_buf_.end(), 0);
    for (const auto& d : doors_) {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int ny = d.y + dy, nx = d.x + dx;
                if (ny >= 0 && ny < grid_size_ && nx >= 0 && nx < grid_size_)
                    near_door_buf_[idx(ny, nx)] = 1;
            }
    }
    for (const auto& [y, x] : empty_cells_) {
        if (near_door_buf_[idx(y, x)]) continue;
        int mask = computeNeighborhoodMask(y, x);
        if (matchesTemplate(y, x, mask))
            candidates.emplace_back(y, x);
    }
    return candidates;
}

// ----------------------------------------------------------------------
// BFS using pre‑allocated buffers
// ----------------------------------------------------------------------
void GridMazeWorld::bfsReachable(int sy, int sx, int maxdist,
                                 const std::vector<uint8_t>& pass_mask,
                                 std::vector<int>& dist, std::vector<int>& queue) {
    std::fill(dist.begin(), dist.end(), -1);
    int head = 0, tail = 0;
    int start = idx(sy, sx);
    if (!pass_mask[start]) return;
    dist[start] = 0;
    queue[tail++] = start;

    while (head < tail) {
        int cur = queue[head++];
        int d = dist[cur];
        if (d >= maxdist) continue;
        int cy = cur / grid_size_, cx = cur % grid_size_;

        if (cy > 0) {
            int nid = idx(cy-1, cx);
            if (pass_mask[nid] && dist[nid] == -1) {
                dist[nid] = d+1;
                queue[tail++] = nid;
            }
        }
        if (cy < grid_size_-1) {
            int nid = idx(cy+1, cx);
            if (pass_mask[nid] && dist[nid] == -1) {
                dist[nid] = d+1;
                queue[tail++] = nid;
            }
        }
        if (cx > 0) {
            int nid = idx(cy, cx-1);
            if (pass_mask[nid] && dist[nid] == -1) {
                dist[nid] = d+1;
                queue[tail++] = nid;
            }
        }
        if (cx < grid_size_-1) {
            int nid = idx(cy, cx+1);
            if (pass_mask[nid] && dist[nid] == -1) {
                dist[nid] = d+1;
                queue[tail++] = nid;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Label connected components – uses reused buffers
// ----------------------------------------------------------------------
void GridMazeWorld::labelConnectedComponents(const std::vector<uint8_t>& pass_mask,
                                             std::vector<int>& labels, int& nlabels) {
    std::fill(labels.begin(), labels.end(), 0);
    nlabels = 0;
    int sp = 0;
    for (int y = 0; y < grid_size_; ++y)
        for (int x = 0; x < grid_size_; ++x) {
            int cid = idx(y, x);
            if (pass_mask[cid] && labels[cid] == 0) {
                ++nlabels;
                int lab = nlabels;
                sp = 0;
                stack_buf_[sp++] = cid;
                labels[cid] = lab;
                while (sp > 0) {
                    int cur = stack_buf_[--sp];
                    int cy = cur / grid_size_, cx = cur % grid_size_;
                    if (cy > 0) {
                        int nid = idx(cy-1, cx);
                        if (pass_mask[nid] && labels[nid] == 0) {
                            labels[nid] = lab;
                            stack_buf_[sp++] = nid;
                        }
                    }
                    if (cy < grid_size_-1) {
                        int nid = idx(cy+1, cx);
                        if (pass_mask[nid] && labels[nid] == 0) {
                            labels[nid] = lab;
                            stack_buf_[sp++] = nid;
                        }
                    }
                    if (cx > 0) {
                        int nid = idx(cy, cx-1);
                        if (pass_mask[nid] && labels[nid] == 0) {
                            labels[nid] = lab;
                            stack_buf_[sp++] = nid;
                        }
                    }
                    if (cx < grid_size_-1) {
                        int nid = idx(cy, cx+1);
                        if (pass_mask[nid] && labels[nid] == 0) {
                            labels[nid] = lab;
                            stack_buf_[sp++] = nid;
                        }
                    }
                }
            }
        }
}

// ----------------------------------------------------------------------
// Check door placement viability
// ----------------------------------------------------------------------
bool GridMazeWorld::canPlaceDoorWithButtons(int y, int x, std::vector<std::pair<int,int>>& btns) {
    if (grid_[idx(y, x)] != static_cast<uint8_t>(TileType::EMPTY)) return false;

    std::fill(pass_buf_.begin(), pass_buf_.end(), 0);
    for (int i = 0; i < total_cells_; ++i) {
        uint8_t t = grid_[i];
        if (t == static_cast<uint8_t>(TileType::EMPTY) ||
            t == static_cast<uint8_t>(TileType::FOOD) ||
            t == static_cast<uint8_t>(TileType::FOOD_SOURCE) ||
            t == static_cast<uint8_t>(TileType::DOOR_OPEN) ||
            t == static_cast<uint8_t>(TileType::BUTTON) ||
            t == static_cast<uint8_t>(TileType::BUTTON_BROKEN))
            pass_buf_[i] = 1;
    }
    pass_buf_[idx(y, x)] = 0;

    int nlabels = 0;
    labelConnectedComponents(pass_buf_, labels_buf_, nlabels);
    if (nlabels < 2) return false;
    if (n_buttons_per_door_ > 0 && nlabels > n_buttons_per_door_) return false;

    btns.clear();
    int maxdist = std::max(0, door_open_duration_ - 2);
    bfsReachable(y, x, maxdist, pass_buf_, dist_buf_, queue_buf_);

    for (int region = 1; region <= nlabels; ++region) {
        std::vector<std::pair<int,int>> candidates;
        for (int i = 0; i < total_cells_; ++i) {
            if (labels_buf_[i] == region && grid_[i] == static_cast<uint8_t>(TileType::EMPTY) && dist_buf_[i] != -1) {
                candidates.emplace_back(i / grid_size_, i % grid_size_);
            }
        }
        if (candidates.empty()) return false;
        std::uniform_int_distribution<int> idxDist(0, candidates.size()-1);
        btns.push_back(candidates[idxDist(rng_)]);
    }
    return true;
}

// ----------------------------------------------------------------------
// Place doors and buttons
// ----------------------------------------------------------------------
void GridMazeWorld::initDoorsAndButtons() {
    doors_.clear();
    buttons_.clear();
    if (n_doors_ == 0) return;
    int placed = 0, attempts = 0;
    const int maxAttempts = 50;
    int nextNumber = 1;
    while (placed < n_doors_ && attempts < maxAttempts) {
        ++attempts;
        auto candidates = findDoorCandidates();
        if (candidates.empty()) break;
        std::shuffle(candidates.begin(), candidates.end(), rng_);
        bool placedAny = false;
        for (auto [y, x] : candidates) {
            if (placed >= n_doors_) break;
            bool tooClose = false;
            for (const auto& d : doors_)
                if (manhattanDistance(y, x, d.y, d.x) < 3) { tooClose = true; break; }
            if (tooClose) continue;
            bool requiresButton = true;
            if (task_class_ == "doors") requiresButton = false;
            else if (task_class_ == "complex") {
                std::uniform_real_distribution<float> prob(0,1);
                requiresButton = prob(rng_) < 0.5f;
            }
            if (!requiresButton) {
                Door d{y, x, door_open_duration_, door_close_duration_, nextNumber,
                       false, true, false, 0};
                std::uniform_real_distribution<float> coin(0,1);
                d.is_open = coin(rng_) < 0.5f;
                doors_.push_back(d);
                grid_[idx(y, x)] = static_cast<uint8_t>(TileType::DOOR_CLOSED);
                door_open_[idx(y, x)] = d.is_open ? 1 : 0;
                ++nextNumber;
                ++placed;
                placedAny = true;
                break;
            } else {
                std::vector<std::pair<int,int>> btns;
                if (canPlaceDoorWithButtons(y, x, btns)) {
                    Door d{y, x, door_open_duration_, door_close_duration_, nextNumber,
                           true, true, false, 0};
                    int doorIdx = static_cast<int>(doors_.size());
                    doors_.push_back(d);
                    grid_[idx(y, x)] = static_cast<uint8_t>(TileType::DOOR_CLOSED);
                    door_open_[idx(y, x)] = 0;
                    for (auto [by, bx] : btns) {
                        Button btn{by, bx, doorIdx, nextNumber, button_break_probability_, false};
                        buttons_.push_back(btn);
                        grid_[idx(by, bx)] = static_cast<uint8_t>(TileType::BUTTON);
                    }
                    ++nextNumber;
                    ++placed;
                    placedAny = true;
                    break;
                }
            }
        }
        if (!placedAny) break;
    }
}

// ----------------------------------------------------------------------
// One‑time initialisation of passable_mask_
// ----------------------------------------------------------------------
void GridMazeWorld::initPassableCache() {
    passable_mask_.resize(total_cells_);
    for (int i = 0; i < total_cells_; ++i) {
        uint8_t t = static_grid_[i];
        bool blocked = (t == static_cast<uint8_t>(TileType::OBSTACLE) ||
                        (t == static_cast<uint8_t>(TileType::DOOR_CLOSED)));
        passable_mask_[i] = blocked ? 0 : 1;
    }
    for (const auto& d : doors_) {
        if (d.is_open) passable_mask_[idx(d.y, d.x)] = 1;
    }
}

// ----------------------------------------------------------------------
// Incremental door state update – only changes one cell when a door toggles
// ----------------------------------------------------------------------
inline void GridMazeWorld::setDoorOpen(int doorIdx, bool open) {
    Door& d = doors_[doorIdx];
    if (d.is_open == open) return;
    d.is_open = open;
    d.timer = 0;
    int cid = idx(d.y, d.x);
    door_open_[cid] = open ? 1 : 0;
    passable_mask_[cid] = open ? 1 : 0;
}

void GridMazeWorld::updateDoorStates() {
    for (int i = 0; i < (int)doors_.size(); ++i) {
        Door& d = doors_[i];
        if (agent_y_ == d.y && agent_x_ == d.x) {
            if (d.is_open) d.timer = 0;
            continue;
        }
        if (d.is_open) {
            if (++d.timer >= d.open_duration) {
                setDoorOpen(i, false);
            }
        } else if (!d.requires_button && d.can_be_opened) {
            if (++d.timer >= d.close_duration) {
                setDoorOpen(i, true);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Manhattan distance
// ----------------------------------------------------------------------
int GridMazeWorld::manhattanDistance(int y1, int x1, int y2, int x2) const {
    return std::abs(y1 - y2) + std::abs(x1 - x2);
}

// ----------------------------------------------------------------------
// Cache spawn cells and coordinate lists for Python bindings
// ----------------------------------------------------------------------
void GridMazeWorld::cacheResetState() {
    spawn_cells_.clear();
    for (int y = 1; y < grid_size_-1; ++y)
        for (int x = 1; x < grid_size_-1; ++x)
            if (static_grid_[idx(y, x)] == static_cast<uint8_t>(TileType::EMPTY))
                spawn_cells_.emplace_back(y, x);

    if (spawn_cells_.empty()) {
        for (int y = 0; y < grid_size_; ++y)
            for (int x = 0; x < grid_size_; ++x)
                if (static_grid_[idx(y, x)] != static_cast<uint8_t>(TileType::OBSTACLE))
                    spawn_cells_.emplace_back(y, x);
    }

    _door_coords.clear();
    for (const auto& d : doors_) _door_coords.emplace_back(d.y, d.x);
    _button_coords.clear();
    for (const auto& b : buttons_) _button_coords.emplace_back(b.y, b.x);
    _food_coords.clear();
    for (const auto& fs : food_sources_) _food_coords.emplace_back(fs.y, fs.x);
}

// ----------------------------------------------------------------------
// Reset environment
// ----------------------------------------------------------------------
std::tuple<std::vector<int>, std::map<std::string, double>> GridMazeWorld::reset(std::optional<int> seed) {
    if (seed) rng_.seed(*seed);

    if (n_food_sources_ > 0) {
        std::fill(food_cell_to_idx_.begin(), food_cell_to_idx_.end(), -1);
        active_depleted_food_.clear();
        std::fill(active_pos_.begin(), active_pos_.end(), -1);
    }

    placeObstaclesWithConnectivity();
    initFoodSources();

    while (!regrow_heap_.empty()) regrow_heap_.pop();
    step_counter_ = 0;

    door_open_.assign(total_cells_, 0);
    button_broken_.assign(total_cells_, 0);
    static_grid_ = grid_;

    initDoorsAndButtons();

    std::fill(button_at_cell_.begin(), button_at_cell_.end(), -1);
    std::fill(door_at_cell_.begin(), door_at_cell_.end(), -1);
    working_buttons_per_door_.assign(doors_.size(), 0);
    for (size_t i = 0; i < doors_.size(); ++i) {
        door_at_cell_[idx(doors_[i].y, doors_[i].x)] = (int)i;
    }
    for (size_t i = 0; i < buttons_.size(); ++i) {
        const auto& b = buttons_[i];
        button_at_cell_[idx(b.y, b.x)] = (int)i;
        if (!b.is_broken) ++working_buttons_per_door_[b.door_idx];
    }

    initPassableCache();
    cacheResetState();

    std::uniform_int_distribution<int> sd(0, static_cast<int>(spawn_cells_.size())-1);
    auto [sy, sx] = spawn_cells_[sd(rng_)];
    agent_y_ = sy; agent_x_ = sx;
    energy_ = initial_energy_;
    steps_ = 0;
    done_ = false;
    last_action_ = 6;
    n_doors_active_ = static_cast<int>(doors_.size());
    n_buttons_working_ = static_cast<int>(buttons_.size());

    std::map<std::string, double> info;
    info["energy"] = energy_;
    info["steps"] = steps_;
    info["complexity_level"] = complexity_level_;
    info["n_doors"] = static_cast<double>(doors_.size());
    info["n_buttons"] = static_cast<double>(buttons_.size());
    info["n_doors_active"] = static_cast<double>(n_doors_active_);
    info["n_buttons_working"] = static_cast<double>(n_buttons_working_);

    std::vector<int> obs_copy = getObservation();
    return {std::move(obs_copy), info};
}

// ----------------------------------------------------------------------
// Soft reset – keep same layout but reposition agent and regrow food
// ----------------------------------------------------------------------
std::tuple<std::vector<int>, std::map<std::string, double>> GridMazeWorld::soft_reset() {
    if (n_food_sources_ > 0) {
        active_depleted_food_.clear();
        std::fill(active_pos_.begin(), active_pos_.end(), -1);
        for (size_t i = 0; i < food_sources_.size(); ++i) {
            auto& fs = food_sources_[i];
            fs.delay = std::uniform_int_distribution<int>(10,30)(rng_);
            fs.exists = 1;
            fs.count = 0;
            food_cache_[idx(fs.y, fs.x)] = 1;
            food_exists_cache_[i] = true;
        }
    }

    while (!regrow_heap_.empty()) regrow_heap_.pop();
    step_counter_ = 0;

    std::uniform_int_distribution<int> sd(0, static_cast<int>(spawn_cells_.size())-1);
    auto [sy, sx] = spawn_cells_[sd(rng_)];
    agent_y_ = sy; agent_x_ = sx;
    energy_ = initial_energy_;
    steps_ = 0;
    done_ = false;
    last_action_ = 6;

    door_open_.assign(total_cells_, 0);
    for (auto& d : doors_) {
        d.is_open = false;
        d.timer = 0;
        d.can_be_opened = true;
    }
    button_broken_.assign(total_cells_, 0);
    for (auto& b : buttons_) {
        b.is_broken = false;
    }
    n_doors_active_ = static_cast<int>(doors_.size());
    n_buttons_working_ = static_cast<int>(buttons_.size());

    std::fill(working_buttons_per_door_.begin(), working_buttons_per_door_.end(), 0);
    for (size_t i = 0; i < buttons_.size(); ++i) {
        if (!buttons_[i].is_broken) ++working_buttons_per_door_[buttons_[i].door_idx];
    }

    initPassableCache();

    std::map<std::string, double> info;
    info["energy"] = energy_;
    info["steps"] = steps_;
    info["complexity_level"] = complexity_level_;
    info["n_doors"] = static_cast<double>(doors_.size());
    info["n_buttons"] = static_cast<double>(buttons_.size());
    info["n_doors_active"] = static_cast<double>(n_doors_active_);
    info["n_buttons_working"] = static_cast<double>(n_buttons_working_);

    std::vector<int> obs_copy = getObservation();
    return {std::move(obs_copy), info};
}

// ----------------------------------------------------------------------
// Press a button – O(1) using button_at_cell_ lookup
// ----------------------------------------------------------------------
bool GridMazeWorld::pressButton(int by, int bx) {
    int cid = idx(by, bx);
    int bi = button_at_cell_[cid];
    if (bi < 0) return false;
    Button& b = buttons_[bi];
    if (b.is_broken) return false;

    if (button_break_probability_ > 0.0f) {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        if (prob(rng_) < button_break_probability_) {
            b.is_broken = true;
            button_broken_[cid] = 1;
            if (--working_buttons_per_door_[b.door_idx] == 0) {
                doors_[b.door_idx].can_be_opened = false;
                --n_doors_active_;
            }
            return false;
        }
    }

    if (b.door_idx < (int)doors_.size() && doors_[b.door_idx].can_be_opened) {
        setDoorOpen(b.door_idx, true);
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------
// Get observation vector – no heap allocation
// ----------------------------------------------------------------------
const std::vector<int>& GridMazeWorld::getObservation() {
    static const int dy[8] = {-1,-1,-1,0,0,1,1,1};
    static const int dx[8] = {-1,0,1,-1,1,-1,0,1};
    for (int i = 0; i < 8; ++i) {
        int ny = agent_y_ + dy[i], nx = agent_x_ + dx[i];
        if (ny < 0 || ny >= grid_size_ || nx < 0 || nx >= grid_size_) {
            obs_buffer_[i] = 1;
            continue;
        }
        int cid = idx(ny, nx);
        if (food_cache_[cid] == 1) {
            obs_buffer_[i] = 3;
            continue;
        }
        uint8_t t = static_grid_[cid];
        if (t == static_cast<uint8_t>(TileType::DOOR_CLOSED))
            obs_buffer_[i] = door_open_[cid] ? 5 : 4;
        else if (t == static_cast<uint8_t>(TileType::BUTTON))
            obs_buffer_[i] = 6;
        else if (t == static_cast<uint8_t>(TileType::FOOD_SOURCE))
            obs_buffer_[i] = 2;
        else if (t == static_cast<uint8_t>(TileType::OBSTACLE))
            obs_buffer_[i] = 1;
        else
            obs_buffer_[i] = 0;
    }
    obs_buffer_[8] = 7 + last_action_;
    int energyLevel = static_cast<int>(energy_ * 0.05f);
    energyLevel = std::clamp(energyLevel, 0, 4);
    obs_buffer_[9] = 14 + energyLevel;
    return obs_buffer_;
}

// ----------------------------------------------------------------------
// Main step function – returns const ref to observation and StepInfo (no map)
// ----------------------------------------------------------------------
std::tuple<const std::vector<int>&, double, bool, bool, StepInfo>
GridMazeWorld::step(int action) {
    if (done_) {
        StepInfo info{};
        info.energy = energy_;
        info.steps = steps_;
        // other fields zero
        return {getObservation(), 0.0, true, true, info};
    }

    updateDoorStates();

    bool buttonPressed = false;
    bool moved = false;
    int y = agent_y_, x = agent_x_;
    int agent_cell = idx(y, x);
    step_counter_++;

    if (action == static_cast<int>(Action::BUTTON)) {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int ny = agent_y_ + dy, nx = agent_x_ + dx;
                if (ny >= 0 && ny < grid_size_ && nx >= 0 && nx < grid_size_ &&
                    static_grid_[idx(ny, nx)] == static_cast<uint8_t>(TileType::BUTTON)) {
                    if (pressButton(ny, nx)) buttonPressed = true;
                }
            }
    } else {
        moved = true;
        switch (static_cast<Action>(action)) {
            case Action::LEFT:  if (x > 0 && canMoveTo(y, x-1)) --x; break;
            case Action::RIGHT: if (x < grid_size_-1 && canMoveTo(y, x+1)) ++x; break;
            case Action::UP:    if (y > 0 && canMoveTo(y-1, x)) --y; break;
            case Action::DOWN:  if (y < grid_size_-1 && canMoveTo(y+1, x)) ++y; break;
            default: break;
        }
        agent_y_ = y; agent_x_ = x;
        agent_cell = idx(y, x);
    }

    float energy_gained = 0.0f;
    if (moved && !food_sources_.empty()) {
        int fi = food_cell_to_idx_[agent_cell];
        if (fi != -1) {
            auto& fs = food_sources_[fi];
            if (fs.exists) {
                fs.exists = 0;
                energy_gained += food_energy_;
                ++fs.count;
                int baseDelay = std::uniform_int_distribution<int>(10,30)(rng_);
                int delay = static_cast<int>(baseDelay * std::pow(1.2f, fs.count));
                fs.delay = delay;
                food_cache_[agent_cell] = 0;
                food_exists_cache_[fi] = false;
                regrow_heap_.emplace(step_counter_ + delay, fi);
            }
        }
    }

    while (!regrow_heap_.empty() && regrow_heap_.top().first <= step_counter_) {
        int idxF = regrow_heap_.top().second;
        regrow_heap_.pop();
        auto& fs = food_sources_[idxF];
        if (!fs.exists) {
            fs.exists = 1;
            food_cache_[idx(fs.y, fs.x)] = 1;
            food_exists_cache_[idxF] = true;
        }
    }

    energy_ = energy_ * energy_decay_ + energy_gained - energy_per_step_;
    energy_ = std::clamp(energy_, 0.0f, 100.0f);
    ++steps_;
    last_action_ = action;
    bool terminated = (steps_ >= max_steps_ || energy_ <= 0);
    done_ = terminated;

    double reward = 0.01;
    if (energy_gained > 0) reward += 1.0;
    if (action == static_cast<int>(Action::BUTTON)) reward += buttonPressed ? 0.5 : -0.1;
    if (energy_ < 10) reward -= 0.1;

    // Build StepInfo – no heap allocation
    StepInfo info;
    info.energy = energy_;
    info.steps = steps_;
    info.food_collected = (energy_gained > 0) ? 1 : 0;
    info.button_pressed = buttonPressed ? 1 : 0;
    info.complexity_level = complexity_level_;
    info.n_doors = static_cast<int>(doors_.size());
    info.n_buttons = static_cast<int>(buttons_.size());
    info.n_doors_active = n_doors_active_;
    info.n_buttons_working = n_buttons_working_;

    last_info_ = info;   // store for later optional access

    return {getObservation(), reward, terminated, false, info};
}

// ----------------------------------------------------------------------
// Convert StepInfo to std::map – call only when needed (e.g., logging)
// ----------------------------------------------------------------------
std::map<std::string, double> GridMazeWorld::info_to_map(const StepInfo& info) const {
    std::map<std::string, double> m;
    m["energy"] = info.energy;
    m["steps"] = static_cast<double>(info.steps);
    m["food_collected"] = static_cast<double>(info.food_collected);
    m["button_pressed"] = static_cast<double>(info.button_pressed);
    m["complexity_level"] = info.complexity_level;
    m["n_doors"] = static_cast<double>(info.n_doors);
    m["n_buttons"] = static_cast<double>(info.n_buttons);
    m["n_doors_active"] = static_cast<double>(info.n_doors_active);
    m["n_buttons_working"] = static_cast<double>(info.n_buttons_working);
    return m;
}

// ----------------------------------------------------------------------
// Render to OpenCV image and return as NumPy array
// ----------------------------------------------------------------------
py::array_t<uint8_t> GridMazeWorld::render(int render_size) {
    int cell_size = std::max(1, render_size / grid_size_);
    int img_h = grid_size_ * cell_size;
    int img_w = grid_size_ * cell_size;
    cv::Mat img(img_h, img_w, CV_8UC3, cv::Scalar(0,0,0));

    auto get_color = [](TileType t, bool door_open, bool button_broken) -> cv::Scalar {
        switch(t) {
            case TileType::EMPTY:        return cv::Scalar(40,40,40);
            case TileType::OBSTACLE:     return cv::Scalar(100,100,100);
            case TileType::FOOD_SOURCE:  return cv::Scalar(10,50,10);
            case TileType::FOOD:         return cv::Scalar(50,200,50);
            case TileType::AGENT:        return cv::Scalar(50,50,200);
            case TileType::DOOR_CLOSED:  return door_open ? cv::Scalar(50,50,50) : cv::Scalar(200,200,200);
            case TileType::DOOR_OPEN:    return cv::Scalar(50,50,50);
            case TileType::BUTTON:       return button_broken ? cv::Scalar(200,0,0) : cv::Scalar(0,0,200);
            case TileType::BUTTON_BROKEN:return cv::Scalar(200,0,0);
            default: return cv::Scalar(0,0,0);
        }
    };

    for (int y = 0; y < grid_size_; ++y)
        for (int x = 0; x < grid_size_; ++x) {
            int cid = idx(y, x);
            TileType tile = static_cast<TileType>(static_grid_[cid]);
            bool door_open_flag = (tile == TileType::DOOR_CLOSED && door_open_[cid] == 1);
            bool button_broken_flag = (tile == TileType::BUTTON && button_broken_[cid] == 1);
            cv::Scalar color = get_color(tile, door_open_flag, button_broken_flag);
            cv::rectangle(img, cv::Rect(x*cell_size, y*cell_size, cell_size, cell_size), color, cv::FILLED);
        }

    for (const auto& fs : food_sources_) {
        int cx = fs.x * cell_size + cell_size/2;
        int cy = fs.y * cell_size + cell_size/2;
        int r = cell_size / 3;
        if (fs.exists) {
            cv::circle(img, cv::Point(cx,cy), r, cv::Scalar(0,255,0), -1);
        } else {
            cv::circle(img, cv::Point(cx,cy), r, cv::Scalar(0,0,0), -1);
            std::string text = std::to_string(fs.delay);
            double fontScale = 0.4 * (cell_size / 30.0);
            int thickness = std::max(1, cell_size / 30);
            int baseline;
            cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
            cv::putText(img, text, cv::Point(cx - ts.width/2, cy + ts.height/2),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);
        }
    }

    for (const auto& d : doors_) {
        int cx = d.x * cell_size + cell_size/2;
        int cy = d.y * cell_size + cell_size/2;
        int r = cell_size / 4;
        cv::Scalar color = d.is_open ? cv::Scalar(50,50,50) : cv::Scalar(200,200,200);
        cv::circle(img, cv::Point(cx,cy), r, color, -1);
        std::string text = std::to_string(d.number);
        double fontScale = 0.6 * (cell_size / 30.0);
        int thickness = std::max(1, cell_size / 30);
        int baseline;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::putText(img, text, cv::Point(cx - ts.width/2, cy + ts.height/2),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), thickness);
    }

    for (const auto& b : buttons_) {
        int cx = b.x * cell_size + cell_size/2;
        int cy = b.y * cell_size + cell_size/2;
        int r = cell_size / 5;
        cv::Scalar color = b.is_broken ? cv::Scalar(200,0,0) : cv::Scalar(0,0,200);
        cv::circle(img, cv::Point(cx,cy), r, color, -1);
        int doorNumber = (b.door_idx < static_cast<int>(doors_.size())) ? doors_[b.door_idx].number : 0;
        std::string text = std::to_string(doorNumber);
        double fontScale = 0.5 * (cell_size / 30.0);
        int thickness = std::max(1, cell_size / 40);
        int baseline;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::putText(img, text, cv::Point(cx - ts.width/2, cy + ts.height/2),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);
    }

    int ax = agent_x_ * cell_size + cell_size/2;
    int ay = agent_y_ * cell_size + cell_size/2;
    int r = cell_size / 2;
    cv::circle(img, cv::Point(ax,ay), r, cv::Scalar(255,255,255), -1);

    std::stringstream info_ss, doors_ss;
    info_ss << std::fixed << std::setprecision(1);
    info_ss << "Energy: " << energy_ << " | Step: " << steps_ << "/" << max_steps_;
    info_ss << " | Task: " << task_class_ << " (Lvl: " << complexity_level_ << ")";
    doors_ss << "Doors: " << doors_.size() << " | Buttons: " << buttons_.size();
    cv::putText(img, info_ss.str(), cv::Point(10,15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,255,255), 1);
    cv::putText(img, doors_ss.str(), cv::Point(10,35), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,255,255), 1);

    std::vector<size_t> shape = { static_cast<size_t>(img_h), static_cast<size_t>(img_w), 3 };
    py::array_t<uint8_t> result(shape);
    memcpy(result.mutable_data(), img.data, img.total() * img.elemSize());
    return result;
}

// ----------------------------------------------------------------------
// Neighborhood mask and template matching
// ----------------------------------------------------------------------
int GridMazeWorld::computeNeighborhoodMask(int y, int x) const {
    static const int dy[9] = {-1,-1,-1, 0,0,0, 1,1,1};
    static const int dx[9] = {-1,0,1, -1,0,1, -1,0,1};
    int mask = 0;
    for (int i = 0; i < 9; ++i) {
        int ny = y + dy[i], nx = x + dx[i];
        if (ny < 0 || ny >= grid_size_ || nx < 0 || nx >= grid_size_)
            mask |= (1 << i);
        else {
            uint8_t t = grid_[idx(ny, nx)];
            if (t == static_cast<uint8_t>(TileType::OBSTACLE) ||
                t == static_cast<uint8_t>(TileType::DOOR_CLOSED) ||
                t == static_cast<uint8_t>(TileType::DOOR_OPEN))
                mask |= (1 << i);
        }
    }
    return mask;
}

bool GridMazeWorld::matchesTemplate(int y, int x, int mask) const {
    if (grid_[idx(y, x)] != static_cast<uint8_t>(TileType::EMPTY)) return false;
    if ((mask >> CENTER_IDX) & 1) return false;
    for (const auto& tmpl : TEMPLATES) {
        bool ok = true;
        for (int i = 0; i < 9; ++i) {
            int8_t v = tmpl[i];
            bool bit = (mask >> i) & 1;
            if (v == 0 && bit) { ok = false; break; }
            if (v == 1 && !bit) { ok = false; break; }
        }
        if (ok) return true;
    }
    return false;
}