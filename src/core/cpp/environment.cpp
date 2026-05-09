// environment.cpp
#include "environment.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>

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

GridMazeWorld::GridMazeWorld(int gs, int ms, int nf, float fe, float ie, float ed, float eps,
                   const std::string& tc, float cl, int nd, int dod, int dcd,
                   int nbpd, float bbp)
    : grid_size_(gs), max_steps_(ms), n_food_sources_(nf), food_energy_(fe),
      initial_energy_(ie), energy_decay_(ed), energy_per_step_(eps),
      task_class_(tc), complexity_level_(cl), n_doors_(nd), door_open_duration_(dod),
      door_close_duration_(dcd), n_buttons_per_door_(nbpd), button_break_probability_(bbp),
      rng_(std::random_device{}()) {
    float obstacle_fraction = 0.15f + cl * 0.15f;
    n_obstacles_ = static_cast<int>((gs-2)*(gs-2) * obstacle_fraction);
    adjustParameters();
}

void GridMazeWorld::adjustParameters() {
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

bool GridMazeWorld::isConnected(const std::unordered_set<std::pair<int,int>, PairHash>& empty) const {
    if (empty.empty()) return true;
    auto start = *empty.begin();
    std::unordered_set<std::pair<int,int>, PairHash> visited;
    std::queue<std::pair<int,int>> q;
    q.push(start);
    visited.insert(start);
    while (!q.empty()) {
        auto [y,x] = q.front(); q.pop();
        for (auto [dy,dx] : {std::make_pair(1,0),{-1,0},{0,1},{0,-1}}) {
            int ny = y+dy, nx = x+dx;
            if (ny>=0 && ny<grid_size_ && nx>=0 && nx<grid_size_) {
                if (empty.count({ny,nx}) && !visited.count({ny,nx})) {
                    visited.insert({ny,nx});
                    q.push({ny,nx});
                }
            }
        }
    }
    return visited.size() == empty.size();
}

void GridMazeWorld::placeObstaclesWithConnectivity() {
    grid_.assign(grid_size_, std::vector<uint8_t>(grid_size_, static_cast<uint8_t>(TileType::EMPTY)));
    for (int i=0; i<grid_size_; ++i) {
        grid_[0][i] = static_cast<uint8_t>(TileType::OBSTACLE);
        grid_[grid_size_-1][i] = static_cast<uint8_t>(TileType::OBSTACLE);
        grid_[i][0] = static_cast<uint8_t>(TileType::OBSTACLE);
        grid_[i][grid_size_-1] = static_cast<uint8_t>(TileType::OBSTACLE);
    }
    if (n_obstacles_ <= 0) return;

    std::unordered_set<std::pair<int,int>, PairHash> emptySet;
    for (int y=0; y<grid_size_; ++y)
        for (int x=0; x<grid_size_; ++x)
            if (grid_[y][x] == static_cast<uint8_t>(TileType::EMPTY))
                emptySet.insert({y,x});

    int placed = 0, attempts = 0;
    const int maxAttempts = 100;
    while (placed < n_obstacles_ && attempts < maxAttempts) {
        attempts++;
        std::vector<std::pair<int,int>> candidates;
        for (int y=1; y<grid_size_-1; ++y)
            for (int x=1; x<grid_size_-1; ++x)
                if (grid_[y][x] == static_cast<uint8_t>(TileType::EMPTY))
                    candidates.emplace_back(y,x);
        if (candidates.empty()) break;
        std::uniform_int_distribution<int> dist(0, candidates.size()-1);
        auto [y,x] = candidates[dist(rng_)];
        grid_[y][x] = static_cast<uint8_t>(TileType::OBSTACLE);
        emptySet.erase({y,x});
        if (isConnected(emptySet))
            placed++;
        else {
            grid_[y][x] = static_cast<uint8_t>(TileType::EMPTY);
            emptySet.insert({y,x});
        }
    }
}

void GridMazeWorld::initFoodSources() {
    std::vector<std::pair<int,int>> emptyCells;
    for (int y=1; y<grid_size_-1; ++y)
        for (int x=1; x<grid_size_-1; ++x)
            if (grid_[y][x] == static_cast<uint8_t>(TileType::EMPTY))
                emptyCells.emplace_back(y,x);
    if (emptyCells.empty()) return;
    std::shuffle(emptyCells.begin(), emptyCells.end(), rng_);
    int n_food = std::min((int)emptyCells.size(), n_food_sources_);
    food_sources_.clear();
    std::uniform_int_distribution<int> delayDist(10,30);
    float centre = (grid_size_-1) * 0.5f;
    std::vector<float> dists;
    for (auto [y,x] : emptyCells)
        dists.push_back(std::abs(y-centre) + std::abs(x-centre));
    std::vector<int> idx(emptyCells.size());
    std::iota(idx.begin(), idx.end(), 0);
    float p = std::max(0.05f, 1.0f - complexity_level_);
    int n_centre = static_cast<int>((1.0f - p) * n_food);
    n_centre = std::max(0, std::min(n_centre, n_food));
    if (n_centre > 0) {
        std::partial_sort(idx.begin(), idx.begin()+n_centre, idx.end(),
            [&](int a, int b) { return dists[a] < dists[b]; });
    }
    std::unordered_set<int> used;
    for (int i=0; i<n_food; ++i) {
        int chosen;
        if (i < n_centre)
            chosen = idx[i];
        else {
            int k = std::max(2, static_cast<int>(std::sqrt(emptyCells.size() / std::max(n_food,1))));
            std::uniform_int_distribution<int> offsetX(0, k-1), offsetY(0, k-1);
            int oy = offsetY(rng_), ox = offsetX(rng_);
            std::vector<int> spreadPool;
            for (size_t j=0; j<emptyCells.size(); ++j) {
                auto [y,x] = emptyCells[j];
                if ((y - oy) % k == 0 && (x - ox) % k == 0 && !used.count(j))
                    spreadPool.push_back(j);
            }
            if (spreadPool.empty()) {
                for (size_t j=0; j<emptyCells.size(); ++j)
                    if (!used.count(j)) spreadPool.push_back(j);
            }
            std::uniform_int_distribution<int> spd(0, spreadPool.size()-1);
            chosen = spreadPool[spd(rng_)];
        }
        used.insert(chosen);
        auto [y,x] = emptyCells[chosen];
        food_sources_.push_back({y, x, delayDist(rng_), 1, 0});
        grid_[y][x] = static_cast<uint8_t>(TileType::FOOD_SOURCE);
    }
    food_cache_.assign(grid_size_, std::vector<int8_t>(grid_size_,0));
    for (auto& fs : food_sources_)
        if (fs.exists) food_cache_[fs.y][fs.x] = 1;
}

int GridMazeWorld::computeNeighborhoodMask(int y, int x) const {
    static const int dy[9] = {-1,-1,-1, 0,0,0, 1,1,1};
    static const int dx[9] = {-1,0,1, -1,0,1, -1,0,1};
    int mask = 0;
    for (int i=0; i<9; ++i) {
        int ny = y + dy[i], nx = x + dx[i];
        if (ny<0 || ny>=grid_size_ || nx<0 || nx>=grid_size_)
            mask |= (1 << i);
        else {
            uint8_t t = static_grid_[ny][nx];
            if (t == static_cast<uint8_t>(TileType::OBSTACLE) ||
                t == static_cast<uint8_t>(TileType::DOOR_CLOSED) ||
                t == static_cast<uint8_t>(TileType::DOOR_OPEN))
                mask |= (1 << i);
        }
    }
    return mask;
}

bool GridMazeWorld::matchesTemplate(int y, int x, int mask) const {
    if (static_grid_[y][x] != static_cast<uint8_t>(TileType::EMPTY)) return false;
    if ((mask >> CENTER_IDX) & 1) return false;
    for (const auto& tmpl : TEMPLATES) {
        bool good = true;
        for (int i=0; i<9; ++i) {
            int8_t v = tmpl[i];
            bool bit = (mask >> i) & 1;
            if (v == 0 && bit) { good = false; break; }
            if (v == 1 && !bit) { good = false; break; }
        }
        if (good) return true;
    }
    return false;
}

std::vector<std::pair<int,int>> GridMazeWorld::findDoorCandidates() {
    std::vector<std::pair<int,int>> candidates;
    std::vector<std::vector<bool>> near_door(grid_size_, std::vector<bool>(grid_size_, false));
    for (int y=0; y<grid_size_; ++y)
        for (int x=0; x<grid_size_; ++x)
            if (static_grid_[y][x] == static_cast<uint8_t>(TileType::DOOR_CLOSED) ||
                static_grid_[y][x] == static_cast<uint8_t>(TileType::DOOR_OPEN)) {
                for (int dy=-1; dy<=1; ++dy)
                    for (int dx=-1; dx<=1; ++dx) {
                        int ny=y+dy, nx=x+dx;
                        if (ny>=0 && ny<grid_size_ && nx>=0 && nx<grid_size_)
                            near_door[ny][nx] = true;
                    }
            }
    for (int y=1; y<grid_size_-1; ++y)
        for (int x=1; x<grid_size_-1; ++x) {
            if (static_grid_[y][x] != static_cast<uint8_t>(TileType::EMPTY)) continue;
            if (near_door[y][x]) continue;
            int mask = computeNeighborhoodMask(y, x);
            if (matchesTemplate(y, x, mask))
                candidates.emplace_back(y,x);
        }
    return candidates;
}

bool GridMazeWorld::canPlaceDoorWithButtons(int y, int x, std::vector<std::pair<int,int>>& btns) {
    if (static_grid_[y][x] != static_cast<uint8_t>(TileType::EMPTY)) return false;
    std::vector<std::vector<bool>> pass(grid_size_, std::vector<bool>(grid_size_, false));
    for (int i=0; i<grid_size_; ++i)
        for (int j=0; j<grid_size_; ++j) {
            uint8_t t = static_grid_[i][j];
            if (t == static_cast<uint8_t>(TileType::EMPTY) ||
                t == static_cast<uint8_t>(TileType::FOOD) ||
                t == static_cast<uint8_t>(TileType::FOOD_SOURCE) ||
                t == static_cast<uint8_t>(TileType::DOOR_OPEN) ||
                t == static_cast<uint8_t>(TileType::BUTTON) ||
                t == static_cast<uint8_t>(TileType::BUTTON_BROKEN))
                pass[i][j] = true;
        }
    pass[y][x] = false;
    std::vector<std::vector<int>> comp(grid_size_, std::vector<int>(grid_size_, 0));
    int compId = 0;
    std::queue<std::pair<int,int>> q;
    for (int i=0; i<grid_size_; ++i)
        for (int j=0; j<grid_size_; ++j) {
            if (pass[i][j] && comp[i][j]==0) {
                ++compId;
                q.push({i,j});
                comp[i][j] = compId;
                while (!q.empty()) {
                    auto [cy,cx] = q.front(); q.pop();
                    for (auto [dy,dx] : {std::make_pair(1,0),{-1,0},{0,1},{0,-1}}) {
                        int ny=cy+dy, nx=cx+dx;
                        if (ny>=0 && ny<grid_size_ && nx>=0 && nx<grid_size_ &&
                            pass[ny][nx] && comp[ny][nx]==0) {
                            comp[ny][nx] = compId;
                            q.push({ny,nx});
                        }
                    }
                }
            }
        }
    int nRegions = compId;
    if (nRegions < 2) return false;
    if (n_buttons_per_door_>0 && nRegions > n_buttons_per_door_) return false;
    btns.clear();
    int maxDist = std::max(0, door_open_duration_ - 2);
    for (int region=1; region<=nRegions; ++region) {
        std::vector<std::pair<int,int>> candidates;
        std::vector<std::vector<int>> dist(grid_size_, std::vector<int>(grid_size_, -1));
        std::queue<std::pair<int,int>> bfsq;
        bfsq.push({y,x});
        dist[y][x] = 0;
        while (!bfsq.empty()) {
            auto [cy,cx] = bfsq.front(); bfsq.pop();
            int d = dist[cy][cx];
            if (d >= maxDist) continue;
            for (auto [dy,dx] : {std::make_pair(1,0),{-1,0},{0,1},{0,-1}}) {
                int ny=cy+dy, nx=cx+dx;
                if (ny>=0 && ny<grid_size_ && nx>=0 && nx<grid_size_ && dist[ny][nx]==-1 &&
                    pass[ny][nx]) {
                    dist[ny][nx] = d+1;
                    bfsq.push({ny,nx});
                }
            }
        }
        for (int i=0; i<grid_size_; ++i)
            for (int j=0; j<grid_size_; ++j) {
                if (comp[i][j]==region && static_grid_[i][j]==static_cast<uint8_t>(TileType::EMPTY) &&
                    dist[i][j]!=-1)
                    candidates.emplace_back(i,j);
            }
        if (candidates.empty()) return false;
        std::uniform_int_distribution<int> idx(0, candidates.size()-1);
        btns.push_back(candidates[idx(rng_)]);
    }
    return true;
}

void GridMazeWorld::initDoorsAndButtons() {
    doors_.clear(); buttons_.clear();
    if (n_doors_ == 0) return;
    std::vector<std::vector<uint8_t>> current = static_grid_;
    int placed = 0, attempts = 0;
    const int maxAttempts = 50;
    int nextNumber = 1;
    while (placed < n_doors_ && attempts < maxAttempts) {
        attempts++;
        auto candidates = findDoorCandidates();
        if (candidates.empty()) break;
        std::shuffle(candidates.begin(), candidates.end(), rng_);
        bool placedAny = false;
        for (auto [y,x] : candidates) {
            if (placed >= n_doors_) break;
            bool tooClose = false;
            for (const auto& d : doors_)
                if (manhattanDistance(y,x, d.y, d.x) < 3) { tooClose = true; break; }
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
                current[y][x] = static_cast<uint8_t>(TileType::DOOR_CLOSED);
                door_open_[y][x] = d.is_open ? 1 : 0;
                nextNumber++;
                placed++;
                placedAny = true;
                break;
            } else {
                std::vector<std::pair<int,int>> btns;
                if (canPlaceDoorWithButtons(y,x,btns)) {
                    Door d{y, x, door_open_duration_, door_close_duration_, nextNumber,
                           true, true, false, 0};
                    int doorIdx = doors_.size();
                    doors_.push_back(d);
                    current[y][x] = static_cast<uint8_t>(TileType::DOOR_CLOSED);
                    door_open_[y][x] = 0;
                    for (auto [by,bx] : btns) {
                        Button btn{by,bx,doorIdx,nextNumber,button_break_probability_,false};
                        buttons_.push_back(btn);
                        current[by][bx] = static_cast<uint8_t>(TileType::BUTTON);
                    }
                    nextNumber++;
                    placed++;
                    placedAny = true;
                    break;
                }
            }
        }
        if (!placedAny) break;
    }
    static_grid_ = current;
}

void GridMazeWorld::cacheResetState() {
    _spawn_cells.clear();
    for (int y=1; y<grid_size_-1; ++y)
        for (int x=1; x<grid_size_-1; ++x)
            if (static_grid_[y][x] == static_cast<uint8_t>(TileType::EMPTY))
                _spawn_cells.emplace_back(y,x);
    if (_spawn_cells.empty()) {
        for (int y=0; y<grid_size_; ++y)
            for (int x=0; x<grid_size_; ++x)
                if (static_grid_[y][x] != static_cast<uint8_t>(TileType::OBSTACLE))
                    _spawn_cells.emplace_back(y,x);
    }
    _door_coords.clear();
    for (const auto& d : doors_)
        _door_coords.emplace_back(d.y, d.x);
    _button_coords.clear();
    for (const auto& b : buttons_)
        _button_coords.emplace_back(b.y, b.x);
}

std::tuple<std::vector<int>, std::map<std::string, double>> GridMazeWorld::reset(std::optional<int> seed) {
    if (seed) rng_.seed(*seed);
    placeObstaclesWithConnectivity();
    initFoodSources();
    door_open_.assign(grid_size_, std::vector<uint8_t>(grid_size_,0));
    button_broken_.assign(grid_size_, std::vector<uint8_t>(grid_size_,0));
    initDoorsAndButtons();
    cacheResetState();
    std::uniform_int_distribution<int> sd(0, _spawn_cells.size()-1);
    auto [sy,sx] = _spawn_cells[sd(rng_)];
    agent_y_ = sy; agent_x_ = sx;
    energy_ = initial_energy_;
    steps_ = 0;
    done_ = false;
    last_action_ = 6; // ENV_ACTIONS_START
    updatePassableCache();
    std::map<std::string, double> info;
    info["energy"] = energy_;
    info["steps"] = steps_;
    info["task_class"] = 0.0;
    info["complexity_level"] = complexity_level_;
    info["n_doors"] = doors_.size();
    info["n_buttons"] = buttons_.size();
    return {getObservation(), info};
}

std::tuple<std::vector<int>, std::map<std::string, double>> GridMazeWorld::soft_reset() {
    std::uniform_int_distribution<int> sd(0, _spawn_cells.size()-1);
    auto [sy,sx] = _spawn_cells[sd(rng_)];
    agent_y_ = sy; agent_x_ = sx;
    energy_ = initial_energy_;
    steps_ = 0;
    done_ = false;
    last_action_ = 6;
    for (auto& fs : food_sources_) {
        std::uniform_int_distribution<int> delayDist(10,30);
        fs.delay = delayDist(rng_);
        fs.exists = 1;
        fs.count = 0;
        food_cache_[fs.y][fs.x] = 1;
    }
    door_open_.assign(grid_size_, std::vector<uint8_t>(grid_size_,0));
    for (auto& d : doors_) {
        d.is_open = false; d.timer = 0; d.can_be_opened = true;
    }
    button_broken_.assign(grid_size_, std::vector<uint8_t>(grid_size_,0));
    for (auto& b : buttons_) {
        b.is_broken = false;
    }
    updatePassableCache();
    std::map<std::string, double> info;
    info["energy"] = energy_;
    info["steps"] = steps_;
    info["task_class"] = 0.0;
    info["complexity_level"] = complexity_level_;
    info["n_doors"] = doors_.size();
    info["n_buttons"] = buttons_.size();
    return {getObservation(), info};
}

void GridMazeWorld::updatePassableCache() {
    passable_mask_.assign(grid_size_, std::vector<uint8_t>(grid_size_,0));
    for (int y=0; y<grid_size_; ++y)
        for (int x=0; x<grid_size_; ++x) {
            uint8_t t = static_grid_[y][x];
            if (t != static_cast<uint8_t>(TileType::OBSTACLE) &&
                (t != static_cast<uint8_t>(TileType::DOOR_CLOSED) || door_open_[y][x]==1))
                passable_mask_[y][x] = 1;
        }
}

bool GridMazeWorld::canMoveTo(int y, int x) const {
    if (y<0 || y>=grid_size_ || x<0 || x>=grid_size_) return false;
    return passable_mask_[y][x] == 1;
}

void GridMazeWorld::updateDoorStates() {
    bool needUpdate = false;
    for (auto& d : doors_) {
        bool old = d.is_open;
        if (agent_y_ == d.y && agent_x_ == d.x) {
            if (d.is_open) d.timer = 0;
        } else {
            if (d.is_open) {
                d.timer++;
                if (d.timer >= d.open_duration) {
                    d.is_open = false;
                    d.timer = 0;
                }
            } else if (!d.requires_button) {
                d.timer++;
                if (d.timer >= d.close_duration) {
                    d.is_open = true;
                    d.timer = 0;
                }
            }
        }
        if (d.is_open != old) {
            door_open_[d.y][d.x] = d.is_open ? 1 : 0;
            needUpdate = true;
        }
    }
    if (needUpdate) updatePassableCache();
}

bool GridMazeWorld::pressButton(int by, int bx) {
    for (auto& b : buttons_) {
        if (b.y == by && b.x == bx) {
            if (b.is_broken) return false;
            if (button_break_probability_ > 0) {
                std::uniform_real_distribution<float> prob(0,1);
                if (prob(rng_) < button_break_probability_) {
                    b.is_broken = true;
                    button_broken_[by][bx] = 1;
                    bool anyWorking = false;
                    for (const auto& other : buttons_)
                        if (other.door_idx == b.door_idx && !other.is_broken)
                            anyWorking = true;
                    if (!anyWorking && b.door_idx < (int)doors_.size())
                        doors_[b.door_idx].can_be_opened = false;
                    return false;
                }
            }
            if (b.door_idx < (int)doors_.size()) {
                if (doors_[b.door_idx].open()) {
                    door_open_[doors_[b.door_idx].y][doors_[b.door_idx].x] = 1;
                    updatePassableCache();
                    return true;
                }
            }
            break;
        }
    }
    return false;
}

std::vector<int> GridMazeWorld::getObservation() {
    std::vector<int> obs(10);
    static const int dy[8] = {-1,-1,-1,0,0,1,1,1};
    static const int dx[8] = {-1,0,1,-1,1,-1,0,1};
    for (int i=0; i<8; ++i) {
        int ny = agent_y_ + dy[i];
        int nx = agent_x_ + dx[i];
        if (ny<0 || ny>=grid_size_ || nx<0 || nx>=grid_size_) {
            obs[i] = 1; // NEIGHBOR_OBSTACLE
            continue;
        }
        if (food_cache_[ny][nx] == 1) {
            obs[i] = 3; // NEIGHBOR_FOOD
            continue;
        }
        uint8_t t = static_grid_[ny][nx];
        if (t == static_cast<uint8_t>(TileType::DOOR_CLOSED)) {
            obs[i] = door_open_[ny][nx] ? 5 : 4;
        } else if (t == static_cast<uint8_t>(TileType::BUTTON)) {
            obs[i] = 6;
        } else if (t == static_cast<uint8_t>(TileType::FOOD_SOURCE)) {
            obs[i] = 2;
        } else if (t == static_cast<uint8_t>(TileType::OBSTACLE)) {
            obs[i] = 1;
        } else {
            obs[i] = 0; // EMPTY
        }
    }
    const int ACTION_BASE = 7;
    obs[8] = ACTION_BASE + last_action_;
    const int ENERGY_BASE = 14;
    int energyLevel = static_cast<int>(energy_ * 0.05f);
    if (energyLevel > 4) energyLevel = 4;
    if (energyLevel < 0) energyLevel = 0;
    obs[9] = ENERGY_BASE + energyLevel;
    return obs;
}

void GridMazeWorld::rebuildPassableMask() {
    updatePassableCache();
}

int GridMazeWorld::manhattanDistance(int ay, int ax, int by, int bx) const {
    return std::abs(ay-by) + std::abs(ax-bx);
}

std::tuple<std::vector<int>, double, bool, bool, std::map<std::string, double>> GridMazeWorld::step(int action) {
    if (done_) return {getObservation(), 0.0, true, true, {}};
    updateDoorStates();
    bool buttonPressed = false;
    bool moved = false;
    int y = agent_y_, x = agent_x_;
    if (action == static_cast<int>(Action::BUTTON)) {
        for (int dy=-1; dy<=1; ++dy)
            for (int dx=-1; dx<=1; ++dx) {
                if (dy==0 && dx==0) continue;
                int ny=agent_y_+dy, nx=agent_x_+dx;
                if (ny>=0 && ny<grid_size_ && nx>=0 && nx<grid_size_ &&
                    static_grid_[ny][nx] == static_cast<uint8_t>(TileType::BUTTON)) {
                    if (pressButton(ny,nx)) buttonPressed=true;
                }
            }
    } else {
        moved = true;
        if (action == static_cast<int>(Action::LEFT) && x>0 && canMoveTo(y, x-1)) x--;
        else if (action == static_cast<int>(Action::RIGHT) && x<grid_size_-1 && canMoveTo(y, x+1)) x++;
        else if (action == static_cast<int>(Action::UP) && y>0 && canMoveTo(y-1, x)) y--;
        else if (action == static_cast<int>(Action::DOWN) && y<grid_size_-1 && canMoveTo(y+1, x)) y++;
        agent_y_ = y; agent_x_ = x;
    }
    float energy_gained = 0.0f;
    if (moved) {
        for (auto& fs : food_sources_) {
            if (fs.y == agent_y_ && fs.x == agent_x_ && fs.exists) {
                fs.exists = 0;
                energy_gained += food_energy_;
                fs.count++;
                int baseDelay = std::uniform_int_distribution<int>(10,30)(rng_);
                fs.delay = static_cast<int>(baseDelay * std::pow(1.2f, fs.count));
                food_cache_[agent_y_][agent_x_] = 0;
            } else if (fs.delay > 0) {
                fs.delay--;
            } else if (fs.delay == 0 && fs.exists == 0) {
                fs.exists = 1;
                food_cache_[fs.y][fs.x] = 1;
            }
        }
    }
    energy_ = energy_ * energy_decay_ + energy_gained - energy_per_step_;
    if (energy_ < 0) energy_ = 0;
    if (energy_ > 100) energy_ = 100;
    steps_++;
    last_action_ = action;
    bool terminated = (steps_ >= max_steps_ || energy_ <= 0);
    bool truncated = false;
    done_ = terminated;
    double reward = 0.01;
    if (energy_gained > 0) reward += 1.0;
    if (action == static_cast<int>(Action::BUTTON)) reward += buttonPressed ? 0.5 : -0.1;
    if (energy_ < 10) reward -= 0.1;
    std::map<std::string, double> info;
    info["energy"] = energy_;
    info["steps"] = steps_;
    info["food_collected"] = (energy_gained>0) ? 1.0 : 0.0;
    info["button_pressed"] = buttonPressed ? 1.0 : 0.0;
    info["task_class"] = 0.0;
    info["complexity_level"] = complexity_level_;
    info["n_doors_active"] = doors_.size();
    info["n_buttons_working"] = buttons_.size();
    return {getObservation(), reward, terminated, truncated, info};
}