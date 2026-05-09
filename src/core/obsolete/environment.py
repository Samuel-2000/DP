# environment.py

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from numba import njit, prange
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import math

from ..constants import (
    OBSERVATION_SIZE, VOCAB_SIZE,
    Actions, NUM_ACTIONS, ENV_ACTIONS_START, ACTION_BOTTOM_VALUE, ENERGY_BOTTOM_VALUE,
    TileType, ObservationTokens, TILE_COLORS,
    TaskClass,
    FOOD_COUNT_MAX, FOOD_COUNT_MIN, MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME,
    FOOD_REGEN_GROWTH_FACTOR,
    FOOD_INTERVAL_INDEX, FOOD_EXISTS_INDEX, FOOD_COLLECTION_COUNT_INDEX
)

# ------------------------------------------------------------
# GLOBAL OFFSETS (module scope) – used in observation
# ------------------------------------------------------------
_OBS_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
_OBS_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)


# ------------------------------------------------------------
# Numba helpers
# ------------------------------------------------------------
@njit(cache=True)
def _label_components_numba_inplace(pass_mask: np.ndarray, labels: np.ndarray):
    h, w = pass_mask.shape
    labels[:] = 0
    nlabels = 0
    maxsize = h * w
    stack_y = np.empty(maxsize, dtype=np.int32)
    stack_x = np.empty(maxsize, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if pass_mask[i, j] == 1 and labels[i, j] == 0:
                nlabels += 1
                lab = nlabels
                top = 0
                stack_y[top] = i
                stack_x[top] = j
                top += 1
                labels[i, j] = lab
                while top > 0:
                    top -= 1
                    cy = stack_y[top]
                    cx = stack_x[top]
                    # up
                    ny = cy - 1
                    if ny >= 0 and pass_mask[ny, cx] == 1 and labels[ny, cx] == 0:
                        labels[ny, cx] = lab
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1
                    # down
                    ny = cy + 1
                    if ny < h and pass_mask[ny, cx] == 1 and labels[ny, cx] == 0:
                        labels[ny, cx] = lab
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1
                    # left
                    nx = cx - 1
                    if nx >= 0 and pass_mask[cy, nx] == 1 and labels[cy, nx] == 0:
                        labels[cy, nx] = lab
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1
                    # right
                    nx = cx + 1
                    if nx < w and pass_mask[cy, nx] == 1 and labels[cy, nx] == 0:
                        labels[cy, nx] = lab
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1
    return nlabels


@njit(cache=True, parallel=True)
def add_obstacles_connectivity(grid: np.ndarray, n_obstacles: int) -> np.ndarray:
    h, w = grid.shape
    total_cells = h * w
    empty_mask = (grid == TileType.EMPTY).flatten()
    empty_ids = np.where(empty_mask)[0]
    n_empty = len(empty_ids)
    if n_obstacles > n_empty - 1:
        n_obstacles = n_empty - 1
    visited = np.zeros(total_cells, dtype=np.uint8)
    queue = np.empty(total_cells, dtype=np.int32)
    added = 0
    for _ in range(n_obstacles):
        for attempt in range(n_empty):
            pick = np.random.randint(0, n_empty)
            cell = empty_ids[pick]
            r, c = cell // w, cell % w
            grid[r, c] = TileType.OBSTACLE
            start = -1
            for j in range(n_empty):
                if j == pick:
                    continue
                nid = empty_ids[j]
                rr, cc = nid // w, nid % w
                if grid[rr, cc] == TileType.EMPTY:
                    start = nid
                    break
            if start < 0:
                grid[r, c] = TileType.EMPTY
                continue
            visited.fill(0)
            head = tail = 0
            visited[start] = 1
            queue[0] = start
            tail = 1
            reach = 1
            while head < tail:
                cur = queue[head]
                head += 1
                cr, cc = cur // w, cur % w
                if cr > 0 and grid[cr-1, cc] == TileType.EMPTY:
                    nid = (cr-1) * w + cc
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                if cr < h-1 and grid[cr+1, cc] == TileType.EMPTY:
                    nid = (cr+1) * w + cc
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                if cc > 0 and grid[cr, cc-1] == TileType.EMPTY:
                    nid = cr * w + (cc-1)
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                if cc < w-1 and grid[cr, cc+1] == TileType.EMPTY:
                    nid = cr * w + (cc+1)
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
            if reach == n_empty - 1:
                empty_ids[pick] = empty_ids[n_empty - 1]
                n_empty -= 1
                added += 1
                break
            else:
                grid[r, c] = TileType.EMPTY
    return grid


@njit(cache=True)
def food_step(
    agent_y: int,
    agent_x: int,
    food_sources: np.ndarray,
    food_energy: float,
    regrown_buffer: np.ndarray
) -> Tuple[float, int]:
    """
    Returns (energy_gained, number_of_regrown_food_sources).
    regrown_buffer must be large enough (size >= number of food sources).
    """
    energy_gained = 0.0
    regrown_count = 0
    n_food = food_sources.shape[0]

    for i in range(n_food):
        y = food_sources[i, 0]
        x = food_sources[i, 1]
        time_left = food_sources[i, FOOD_INTERVAL_INDEX]
        has_food = food_sources[i, FOOD_EXISTS_INDEX]
        collect_cnt = food_sources[i, FOOD_COLLECTION_COUNT_INDEX]

        if agent_y == y and agent_x == x and has_food == 1:
            food_sources[i, FOOD_EXISTS_INDEX] = 0
            energy_gained += food_energy
            new_cnt = collect_cnt + 1
            food_sources[i, FOOD_COLLECTION_COUNT_INDEX] = new_cnt
            base_regen = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME)
            new_delay = base_regen * (FOOD_REGEN_GROWTH_FACTOR ** new_cnt)
            food_sources[i, FOOD_INTERVAL_INDEX] = int(new_delay)
        elif time_left > 0:
            food_sources[i, FOOD_INTERVAL_INDEX] = time_left - 1
        elif time_left == 0:
            food_sources[i, FOOD_EXISTS_INDEX] = 1
            regrown_buffer[regrown_count] = i
            regrown_count += 1

    return energy_gained, regrown_count


@njit(cache=True)
def get_observation_optimized(
    y: int,
    x: int,
    static_grid: np.ndarray,
    last_action: int,
    energy: float,
    food_positions_cache: np.ndarray,
    door_open_array: np.ndarray,
    button_broken_array: np.ndarray
) -> np.ndarray:
    obs = np.empty(10, dtype=np.int32)
    h = static_grid.shape[0]
    w = static_grid.shape[1]

    for i in range(8):
        ny = y + _OBS_DY[i]
        nx = x + _OBS_DX[i]
        if ny < 0 or ny >= h or nx < 0 or nx >= w:
            obs[i] = ObservationTokens.NEIGHBOR_OBSTACLE
            continue
        if food_positions_cache[ny, nx] > 0:
            obs[i] = ObservationTokens.NEIGHBOR_FOOD
            continue
        tile = static_grid[ny, nx]
        if tile == TileType.DOOR_CLOSED:
            if door_open_array[ny, nx] == 1:
                obs[i] = ObservationTokens.NEIGHBOR_DOOR_OPEN
            else:
                obs[i] = ObservationTokens.NEIGHBOR_DOOR_CLOSED
        elif tile == TileType.BUTTON:
            obs[i] = ObservationTokens.NEIGHBOR_BUTTON
        else:
            obs[i] = tile

    obs[8] = ACTION_BOTTOM_VALUE + last_action
    energy_scaled = int(energy * 0.05)   # 0‑100 → 0‑4, faster than division
    if energy_scaled < 0:
        energy_scaled = 0
    elif energy_scaled > 4:
        energy_scaled = 4
    obs[9] = ENERGY_BOTTOM_VALUE + energy_scaled
    return obs


@njit(cache=True)
def bfs_reachable_mask(passable_mask: np.ndarray, h: int, w: int,
                       sy: int, sx: int, maxdist: int) -> np.ndarray:
    visited = np.zeros((h, w), dtype=np.uint8)
    qy = np.empty(h * w, dtype=np.int32)
    qx = np.empty(h * w, dtype=np.int32)
    qd = np.empty(h * w, dtype=np.int32)
    head = tail = 0
    if passable_mask[sy, sx] == 0:
        return visited
    visited[sy, sx] = 1
    qy[tail] = sy
    qx[tail] = sx
    qd[tail] = 0
    tail += 1
    while head < tail:
        cy = qy[head]
        cx = qx[head]
        cd = qd[head]
        head += 1
        if cd >= maxdist:
            continue
        # up
        ny = cy - 1
        nx = cx
        if ny >= 0:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
        # down
        ny = cy + 1
        nx = cx
        if ny < h:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
        # left
        ny = cy
        nx = cx - 1
        if nx >= 0:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
        # right
        ny = cy
        nx = cx + 1
        if nx < w:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
    return visited


# ------------------------------------------------------------
# Data classes (unchanged)
# ------------------------------------------------------------
@dataclass
class Door:
    y: int
    x: int
    open_duration: int
    close_duration: int
    can_be_opened: bool
    requires_button: bool
    is_choke_point: bool
    door_number: int
    is_open: bool = False
    timer: int = 0

    def update(self, agent_pos: Optional[Tuple[int, int]] = None):
        if agent_pos is not None:
            agent_y, agent_x = agent_pos
            if agent_y == self.y and agent_x == self.x:
                if self.is_open:
                    self.timer = 0
                return
        if self.is_open:
            self.timer += 1
            if self.timer >= self.open_duration:
                self.is_open = False
                self.timer = 0
        elif not self.requires_button:
            self.timer += 1
            if self.timer >= self.close_duration:
                self.is_open = True
                self.timer = 0

    def open(self):
        if self.can_be_opened:
            self.is_open = True
            self.timer = 0
            return True
        return False


@dataclass
class Button:
    y: int
    x: int
    door_idx: int
    break_probability: float
    button_number: int
    is_broken: bool = False

    def press(self):
        if self.is_broken:
            return False
        if self.break_probability > 0.0 and np.random.random() < self.break_probability:
            self.is_broken = True
            return False
        return True


# ------------------------------------------------------------
# Template matching (shared instance)
# ------------------------------------------------------------
class TemplateNode:
    __slots__ = ('split_pos', 'pass_child', 'obs_child', 'templates', 'is_leaf')
    def __init__(self, is_leaf: bool, split_pos: int = -1):
        self.is_leaf = is_leaf
        self.split_pos = split_pos
        self.pass_child = None
        self.obs_child = None
        self.templates: List[int] = []


class FastTemplateMatcher:
    _OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 0), (0, 1),
                (1, -1), (1, 0), (1, 1)]
    _CENTER_IDX = 4

    def __init__(self, templates_flat: List[np.ndarray], max_depth: int):
        self.templates_flat = [np.array(t, dtype=np.int8).reshape(9,) for t in templates_flat]
        self.n_templates = len(self.templates_flat)
        self.template_array = np.stack(self.templates_flat, axis=0) if self.n_templates > 0 else np.zeros((0, 9), dtype=np.int8)
        self.obstacle_masks = np.zeros(self.n_templates, dtype=np.uint16)
        self.passable_masks = np.zeros(self.n_templates, dtype=np.uint16)
        for i in range(self.n_templates):
            obs = 0
            pas = 0
            flat = self.template_array[i]
            for j in range(9):
                v = int(flat[j])
                if v == 1:
                    obs |= (1 << j)
                elif v == 0:
                    pas |= (1 << j)
            self.obstacle_masks[i] = obs
            self.passable_masks[i] = pas
        self.root = self._build_tree(list(range(self.n_templates)), depth=0, max_depth=max_depth)

    def compute_all_neighborhood_masks(self, grid: np.ndarray) -> np.ndarray:
        H, W = grid.shape
        obstacle_bool = ((grid == TileType.OBSTACLE) | (grid == TileType.DOOR_CLOSED) | (grid == TileType.DOOR_OPEN))
        p = np.pad(obstacle_bool.astype(np.uint8), pad_width=1, constant_values=1)
        masks = np.zeros((H, W), dtype=np.uint16)
        for bit, (dy, dx) in enumerate(self._OFFSETS):
            sub = p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W].astype(np.uint16)
            masks |= (sub << bit)
        return masks

    def _neighborhood_mask(self, grid: np.ndarray, y: int, x: int) -> int:
        H, W = grid.shape
        mask = 0
        for bit, (dy, dx) in enumerate(self._OFFSETS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                mask |= (1 << bit)
            else:
                t = grid[ny, nx]
                if t == TileType.OBSTACLE or t == TileType.DOOR_CLOSED or t == TileType.DOOR_OPEN:
                    mask |= (1 << bit)
        return mask

    def _entropy_score(self, indices: List[int], pos: int) -> float:
        counts = np.zeros(3, dtype=np.int32)
        for idx in indices:
            val = int(self.template_array[idx, pos]) + 1
            counts[val] += 1
        total = len(indices)
        if total <= 1:
            return 0.0
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy

    def _build_tree(self, indices: List[int], depth: int, max_depth: int) -> TemplateNode:
        if len(indices) <= 2 or depth >= max_depth:
            leaf = TemplateNode(is_leaf=True)
            leaf.templates = indices.copy()
            return leaf
        best_pos = -1
        best_entropy = -1.0
        for pos in range(9):
            if pos == self._CENTER_IDX:
                continue
            entropy = self._entropy_score(indices, pos)
            if entropy > best_entropy:
                best_entropy = entropy
                best_pos = pos
        if best_pos == -1 or best_entropy < 0.08:
            leaf = TemplateNode(is_leaf=True)
            leaf.templates = indices.copy()
            return leaf
        pass_indices: List[int] = []
        obs_indices: List[int] = []
        for idx in indices:
            v = int(self.template_array[idx, best_pos])
            if v == 0 or v == -1:
                pass_indices.append(idx)
            if v == 1 or v == -1:
                obs_indices.append(idx)
        if len(pass_indices) == len(indices) and len(obs_indices) == len(indices):
            leaf = TemplateNode(is_leaf=True)
            leaf.templates = indices.copy()
            return leaf
        node = TemplateNode(is_leaf=False)
        node.split_pos = best_pos
        node.pass_child = self._build_tree(pass_indices, depth + 1, max_depth) if pass_indices else None
        node.obs_child = self._build_tree(obs_indices, depth + 1, max_depth) if obs_indices else None
        return node

    def matches(self, grid: np.ndarray, y: int, x: int, neighborhood_mask: Optional[int] = None) -> bool:
        if grid[y, x] != TileType.EMPTY:
            return False
        if neighborhood_mask is None:
            neighborhood_mask = self._neighborhood_mask(grid, y, x)
        if ((neighborhood_mask >> self._CENTER_IDX) & 1) != 0:
            return False
        node = self.root
        while node is not None and not node.is_leaf:
            bit = (neighborhood_mask >> node.split_pos) & 1
            if bit:
                node = node.obs_child
            else:
                node = node.pass_child
            if node is None:
                break
        if node is None:
            return False
        for ti in node.templates:
            req_obs = int(self.obstacle_masks[ti])
            req_pass = int(self.passable_masks[ti])
            if (neighborhood_mask & req_obs) != req_obs:
                continue
            if (neighborhood_mask & req_pass) != 0:
                continue
            return True
        return False


# ------------------------------------------------------------
# GridMazeWorld
# ------------------------------------------------------------
class GridMazeWorld(gym.Env):
    _ring_offsets_cache = {}
    _shared_template_matcher = None
    _shared_templates_list = None

    @classmethod
    def _get_templates_list(cls):
        if cls._shared_templates_list is None:
            templates_3x3 = [
                np.array([[-1,  0, -1], [ 1,  0,  1], [-1,  0, -1]], dtype=np.int8),
                np.array([[-1,  1, -1], [ 0,  0,  0], [-1,  1, -1]], dtype=np.int8),
                np.array([[-1,  0,  1], [ 0,  0,  0], [ 1,  0, -1]], dtype=np.int8),
                np.array([[ 1,  0, -1], [ 0,  0,  0], [-1,  0,  1]], dtype=np.int8),
                np.array([[-1,  1, -1], [ 0,  0, -1], [ 1,  0, -1]], dtype=np.int8),
                np.array([[-1, -1, -1], [ 0,  0,  1], [ 1,  0, -1]], dtype=np.int8),
                np.array([[-1,  0,  1], [ 1,  0,  0], [-1, -1, -1]], dtype=np.int8),
                np.array([[-1,  0,  1], [-1,  0,  0], [-1,  1, -1]], dtype=np.int8),
                np.array([[-1,  1, -1], [-1,  0,  0], [-1,  0,  1]], dtype=np.int8),
                np.array([[-1, -1, -1], [ 1,  0,  0], [-1,  0,  1]], dtype=np.int8),
                np.array([[ 1,  0, -1], [ 0,  0, -1], [-1,  1, -1]], dtype=np.int8),
                np.array([[ 1,  0, -1], [ 0,  0,  1], [-1, -1, -1]], dtype=np.int8)
            ]
            cls._shared_templates_list = [t.flatten() for t in templates_3x3]
        return cls._shared_templates_list

    @classmethod
    def get_template_matcher(cls, max_depth=4):
        if cls._shared_template_matcher is None:
            cls._shared_template_matcher = FastTemplateMatcher(cls._get_templates_list(), max_depth)
        return cls._shared_template_matcher

    @classmethod
    def _get_ring_offsets(cls, grid_size: int):
        if grid_size not in cls._ring_offsets_cache:
            max_dist = 2 * (grid_size - 1)
            ring_offsets = [[] for _ in range(max_dist + 1)]
            ring_offsets[0] = [(0, 0)]
            for d in range(1, max_dist + 1):
                offsets = set()
                for dy in range(-d, d + 1):
                    dx = d - abs(dy)
                    if dx == 0:
                        offsets.add((dy, 0))
                    else:
                        offsets.add((dy, dx))
                        offsets.add((dy, -dx))
                ring_offsets[d] = list(offsets)
            cls._ring_offsets_cache[grid_size] = ring_offsets
        return cls._ring_offsets_cache[grid_size]

    def __init__(self, grid_size: int, max_steps: int,
                 n_food_sources: int, food_energy: float, initial_energy: float,
                 energy_decay: float, energy_per_step: float,
                 render_size: int, task_class: str, complexity_level: float,
                 n_doors: int, door_open_duration: int, door_close_duration: int,
                 n_buttons_per_door: int, button_break_probability: float):
        super().__init__()
        self.grid_size = grid_size
        self._ring_offsets = self._get_ring_offsets(self.grid_size)
        self.max_steps = max_steps
        self.task_class = task_class
        self.complexity_level = max(0.0, min(1.0, complexity_level))
        
        # ----- Obstacle count: scales linearly with complexity (0.25 → 0.75) -----
        obstacle_fraction = 0.15 + self.complexity_level * 0.15
        self.n_obstacles = int((grid_size - 2) ** 2 * obstacle_fraction)
        
        # ----- Food sources: more complex → fewer food sources -----
        p = max(0.05, 1.0 - self.complexity_level)
        extra_max = FOOD_COUNT_MAX - FOOD_COUNT_MIN
        extra = np.random.binomial(extra_max, p)
        self.n_food_sources = FOOD_COUNT_MIN + extra
        
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size

        self.door_open_duration = door_open_duration
        self.door_close_duration = door_close_duration
        self.n_doors = n_doors
        self.n_buttons_per_door = n_buttons_per_door
        self.button_break_probability = button_break_probability
        self._adjust_parameters_by_task_class()

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=VOCAB_SIZE-1, shape=(OBSERVATION_SIZE,), dtype=np.int32)

        # Mutable state
        self.grid = None
        self.static_grid = None
        self.food_sources = None
        self.food_positions_cache = None
        self.door_open_array = None
        self.button_broken_array = None

        # Agent position (scalars, not numpy array)
        self.agent_y = 0
        self.agent_x = 0

        self.energy = None
        self.steps = None
        self.done = None
        self.last_action = None
        self.doors: List[Door] = []
        self.buttons: List[Button] = []

        # Cached data for soft reset
        self._empty_cells: List[Tuple[int, int]] = []
        self._food_coords: List[Tuple[int, int]] = []
        self._spawn_cells = None
        self._door_coords = None
        self._button_coords = None
        self._info = {}
        self._regen_buffer = np.empty(FOOD_COUNT_MAX, dtype=np.int32)

        # Template matcher (shared)
        self.template_matcher = self.get_template_matcher(max_depth=4)
        self._door_check_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self._passable_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._labels = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.colors = TILE_COLORS
        self.debug = False
        self._passable_cache = None

        # Cached counters
        self._n_doors_active = 0
        self._n_buttons_working = 0

    def _adjust_parameters_by_task_class(self):
        if self.task_class == TaskClass.BASIC:
            self.n_doors = 0
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
        elif self.task_class == TaskClass.DOORS:
            if self.n_doors is None:
                self.n_doors = max(1, int(self.complexity_level * 3))
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
        elif self.task_class == TaskClass.BUTTONS:
            if self.n_doors is None:
                self.n_doors = max(1, int(self.complexity_level * 3))
            if self.n_buttons_per_door is None:
                self.n_buttons_per_door = 4
            if self.button_break_probability is None:
                self.button_break_probability = self.complexity_level * 0.2
        elif self.task_class == TaskClass.COMPLEX:
            if self.n_doors is None:
                self.n_doors = max(2, int(self.complexity_level * 4))
            if self.n_buttons_per_door is None:
                self.n_buttons_per_door = 4
            if self.button_break_probability is None:
                self.button_break_probability = self.complexity_level * 0.3


    """
    # fast food source generation function, but invariant to complexity
    def _init_food_sources(self): # original simple random (fast, but invariant to complexity)
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        if len(empty_cells) == 0 or self.n_food_sources <= 0:
            self.food_sources = np.zeros((0, 4), dtype=np.int32)
            return
        indices = np.random.choice(len(empty_cells), min(self.n_food_sources, len(empty_cells)), replace=False)
        self.food_sources = np.zeros((len(indices), 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME)
            self.food_sources[i] = [y, x, regen_time, 1]
            self.grid[y, x] = TileType.FOOD_SOURCE
    """

    """
    # just a placeholder to understand the idea of the function below
    def _init_food_sources(self): # based on random 1D gaussian sampling and repulsion to make food sources less clustered in levels with high complexity and vice versa (slow)
        empty_cells = [tuple(cell) for cell in np.argwhere(self.grid == TileType.EMPTY)]
        if len(empty_cells) == 0 or self.n_food_sources <= 0:
            self.food_sources = np.zeros((0, 4), dtype=np.int32)
            return

        self.food_sources = np.zeros((self.n_food_sources, 4), dtype=np.int32)
        center = self.grid_size // 2

        # Gaussian distribution for row/col sampling
        std = (1.0 + self.complexity_level * (center - 1))
        indices = np.arange(self.grid_size)
        probs = np.exp(-0.5 * ((indices - center) / std) ** 2)
        probs /= probs.sum()

        empty_set = set(empty_cells)
        size = self.grid_size
        ring_offsets = self._ring_offsets

        # Repulsion strength scales with complexity (0 = none, 1 = full)
        strength = (0.1 + self.complexity_level) * size

        for i in range(self.n_food_sources):
            row = np.random.choice(size, p=probs)
            col = np.random.choice(size, p=probs)

            # Single-step repulsion from already placed foods
            dx_total = 0.0
            dy_total = 0.0
            for j in range(i):
                other_y = self.food_sources[j, 0]
                other_x = self.food_sources[j, 1]
                dy = row - other_y
                dx = col - other_x
                dist = abs(dy) + abs(dx)
                if dist == 0:
                    # Random push if exactly overlapping
                    dx_total += np.random.uniform(-1, 1) * strength
                    dy_total += np.random.uniform(-1, 1) * strength
                else:
                    # Force = strength / (dist + ε) ; direction away
                    force = strength / (dist + 1e-6)
                    dx_total += force * (dx / dist)
                    dy_total += force * (dy / dist)

            # Apply displacement once
            row = int(np.clip(row + dy_total, 0, size - 1))
            col = int(np.clip(col + dx_total, 0, size - 1))

            #print(f"dx_total: {int(dx_total)}, dy_total: {int(dy_total)}, row: {row}, col: {col}")

            # Find nearest empty cell (spiral search)
            found = False
            for d in range(len(ring_offsets)):
                for dy, dx in ring_offsets[d]:
                    ny, nx = row + dy, col + dx
                    if 0 <= ny < size and 0 <= nx < size and (ny, nx) in empty_set:
                        row, col = ny, nx
                        found = True
                        break
                if found:
                    break
            if not found:
                row, col = next(iter(empty_set))

            regen = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME)
            self.food_sources[i] = [row, col, regen, 1]
            self.grid[row, col] = TileType.FOOD_SOURCE
            empty_set.remove((row, col))

        self._update_food_cache()
    """


    # ---------- Generation helpers (unchanged except using scalars) ----------
    def _init_food_sources(self):
        # (unchanged – uses self.grid, self.food_sources, etc.)
        rng = np.random
        n_food = self.n_food_sources
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        N = len(empty_cells)
        if N == 0 or n_food <= 0:
            self.food_sources = np.zeros((0, 5), dtype=np.int32)
            return
        n_food = min(n_food, N)
        size = self.grid_size
        centre = (size - 1) * 0.5
        ec = empty_cells.astype(np.float32)
        dist = np.abs(ec[:, 0] - centre) + np.abs(ec[:, 1] - centre)
        centre_count = min(N, max(n_food, N // 4))
        centre_pool = np.argpartition(dist, centre_count - 1)[:centre_count]
        rng.shuffle(centre_pool)
        k = max(2, int(np.sqrt(N / max(n_food, 1))))
        oy = rng.randint(0, k)
        ox = rng.randint(0, k)
        spread_mask = ((empty_cells[:, 0] - oy) % k == 0) & ((empty_cells[:, 1] - ox) % k == 0)
        spread_pool = np.flatnonzero(spread_mask)
        rng.shuffle(spread_pool)
        c = float(self.complexity_level)
        n_centre = int((1.0 - c) * n_food)
        n_centre = max(0, min(n_centre, n_food))
        chosen = np.empty(n_food, dtype=np.int32)
        used = np.zeros(N, dtype=bool)
        pos = 0
        if n_centre > 0:
            centre_part = centre_pool[:n_centre]
            chosen[:len(centre_part)] = centre_part
            used[centre_part] = True
            pos = len(centre_part)
        if pos < n_food:
            spread_avail = spread_pool[~used[spread_pool]]
            take = min(n_food - pos, len(spread_avail))
            if take > 0:
                part = spread_avail[:take]
                chosen[pos:pos+take] = part
                used[part] = True
                pos += take
        if pos < n_food:
            remaining = np.flatnonzero(~used)
            extra = rng.choice(remaining, size=n_food - pos, replace=False)
            chosen[pos:] = extra
        self.food_sources = np.zeros((n_food, 5), dtype=np.int32)
        regen = rng.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME, size=n_food)
        for i, idx in enumerate(chosen):
            y, x = empty_cells[idx]
            self.food_sources[i] = [y, x, regen[i], 1, 0]
            self.grid[y, x] = TileType.FOOD_SOURCE
        self._update_food_cache()

    def _update_food_cache(self):
        if self.food_sources is None or self.food_sources.shape[0] == 0:
            return
        self.food_positions_cache.fill(0)
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food, _ = self.food_sources[i]
            if has_food:
                self.food_positions_cache[y, x] = 1

    def _find_regions_separated_by_door(self, door_y: int, door_x: int, grid_to_use: np.ndarray) -> List[List[Tuple[int, int]]]:
        h, w = self.grid_size, self.grid_size
        pass_mask = self._passable_mask
        labels = self._labels
        pass_mask[:] = 0
        pass_mask[(grid_to_use == TileType.EMPTY)] = 1
        pass_mask[(grid_to_use == TileType.FOOD)] = 1
        pass_mask[(grid_to_use == TileType.FOOD_SOURCE)] = 1
        pass_mask[(grid_to_use == TileType.DOOR_OPEN)] = 1
        pass_mask[(grid_to_use == TileType.BUTTON)] = 1
        pass_mask[(grid_to_use == TileType.BUTTON_BROKEN)] = 1
        pass_mask[door_y, door_x] = 0
        nlabels = _label_components_numba_inplace(pass_mask, labels)
        regions: List[List[Tuple[int, int]]] = [[] for _ in range(nlabels)]
        for y in range(h):
            for x in range(w):
                lab = labels[y, x]
                if lab > 0:
                    regions[lab - 1].append((y, x))
        return regions

    def _can_place_door_with_buttons(self, y: int, x: int, grid_to_use: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        if grid_to_use[y, x] != TileType.EMPTY:
            return False, []
        regions = self._find_regions_separated_by_door(y, x, grid_to_use)
        required_buttons = len(regions)
        if self.n_buttons_per_door > 0 and required_buttons > self.n_buttons_per_door:
            return False, []
        button_positions: List[Tuple[int, int]] = []
        h, w = grid_to_use.shape
        max_dist = max(0, self.door_open_duration - 2)
        if grid_to_use is self.grid:
            if self._passable_mask is None:
                self._update_passable_mask()
            pass_mask = self._passable_mask
        else:
            pm = np.zeros_like(grid_to_use, dtype=np.uint8)
            pm[np.where((grid_to_use == TileType.EMPTY) |
                        (grid_to_use == TileType.FOOD) |
                        (grid_to_use == TileType.FOOD_SOURCE) |
                        (grid_to_use == TileType.BUTTON) |
                        (grid_to_use == TileType.BUTTON_BROKEN) |
                        (grid_to_use == TileType.DOOR_OPEN))] = 1
            pass_mask = pm
        for i, region in enumerate(regions):
            reachable = bfs_reachable_mask(pass_mask, h, w, y, x, max_dist)
            candidate_positions = []
            for (ry, rx) in region:
                if grid_to_use[ry, rx] == TileType.EMPTY and reachable[ry, rx] == 1:
                    candidate_positions.append((ry, rx))
            if not candidate_positions:
                return False, []
            idx = np.random.randint(0, len(candidate_positions))
            by, bx = candidate_positions[idx]
            button_positions.append((by, bx))
        return True, button_positions

    def _find_door_candidates_with_templates(self, grid_to_use: np.ndarray) -> List[Tuple[int, int]]:
        H, W = self.grid_size, self.grid_size
        masks = self.template_matcher.compute_all_neighborhood_masks(grid_to_use)
        door_bool = ((grid_to_use == TileType.DOOR_OPEN) | (grid_to_use == TileType.DOOR_CLOSED)).astype(np.uint8)
        pdoor = np.pad(door_bool, pad_width=1, constant_values=0)
        near_door = np.zeros((H, W), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                near_door |= pdoor[1 + dy:1 + dy + H, 1 + dx:1 + dx + W].astype(bool)
        ys, xs = np.where((grid_to_use == TileType.EMPTY) & (~near_door))
        candidates: List[Tuple[int, int]] = []
        for y, x in zip(ys, xs):
            m = int(masks[y, x])
            if ((m >> 4) & 1) != 0:
                continue
            if self.template_matcher.matches(grid_to_use, y, x, neighborhood_mask=m):
                candidates.append((int(y), int(x)))
        return candidates

    def _init_doors_and_buttons(self):
        self.doors = []
        self.buttons = []
        if self.n_doors == 0:
            return
        current_grid = self.grid.copy()
        placed_doors = 0
        attempts = 0
        max_attempts = 50
        next_door_number = 1
        while placed_doors < self.n_doors and attempts < max_attempts:
            attempts += 1
            candidates = self._find_door_candidates_with_templates(current_grid)
            if not candidates:
                break
            np.random.shuffle(candidates)
            door_placed_this_round = False
            for y, x in candidates:
                if placed_doors >= self.n_doors:
                    break
                too_close = any(self._manhattan_distance(y, x, d.y, d.x) < 3 for d in self.doors)
                if too_close:
                    continue
                requires_button = True
                if self.task_class == TaskClass.DOORS:
                    requires_button = False
                elif self.task_class == TaskClass.COMPLEX:
                    requires_button = np.random.random() < 0.5
                if not requires_button:
                    door = Door(y=y, x=x, open_duration=self.door_open_duration,
                                close_duration=self.door_close_duration,
                                requires_button=False, can_be_opened=True,
                                is_choke_point=True, door_number=next_door_number)
                    door.is_open = np.random.random() < 0.5
                    self.doors.append(door)
                    self.grid[y, x] = TileType.DOOR_CLOSED
                    current_grid[y, x] = TileType.DOOR_CLOSED
                    self.door_open_array[y, x] = 1 if door.is_open else 0
                    next_door_number += 1
                    placed_doors += 1
                    door_placed_this_round = True
                    break
                else:
                    can_place, button_positions = self._can_place_door_with_buttons(y, x, current_grid)
                    if can_place:
                        door = Door(y=y, x=x, open_duration=self.door_open_duration,
                                    close_duration=self.door_close_duration,
                                    requires_button=True, can_be_opened=True,
                                    is_choke_point=True, door_number=next_door_number)
                        door_idx = len(self.doors)
                        self.doors.append(door)
                        self.grid[y, x] = TileType.DOOR_CLOSED
                        current_grid[y, x] = TileType.DOOR_CLOSED
                        self.door_open_array[y, x] = 0
                        for by, bx in button_positions:
                            button = Button(y=by, x=bx, door_idx=door_idx,
                                            break_probability=self.button_break_probability,
                                            is_broken=False, button_number=next_door_number)
                            self.buttons.append(button)
                            self.grid[by, bx] = TileType.BUTTON
                            current_grid[by, bx] = TileType.BUTTON
                        next_door_number += 1
                        placed_doors += 1
                        door_placed_this_round = True
                        break
            if not door_placed_this_round:
                break

    def _update_passable_mask(self):
        g = self.grid
        mask = np.zeros_like(g, dtype=np.uint8)
        mask[np.where((g == TileType.EMPTY) |
                      (g == TileType.FOOD) |
                      (g == TileType.FOOD_SOURCE) |
                      (g == TileType.BUTTON) |
                      (g == TileType.BUTTON_BROKEN) |
                      (g == TileType.DOOR_OPEN))] = 1
        self._passable_mask = mask

    def _manhattan_distance(self, a_y: int, a_x: int, b_y: int, b_x: int) -> int:
        return abs(a_y - b_y) + abs(a_x - b_x)

    def _cache_reset_state(self):
        self.static_grid = self.grid.copy()
        spawn_cells = np.argwhere(self.static_grid == TileType.EMPTY)
        if len(spawn_cells) == 0:
            spawn_cells = np.argwhere(self.static_grid != TileType.OBSTACLE)
        self._spawn_cells = spawn_cells.astype(np.int32, copy=False)

        if self.food_sources is not None and self.food_sources.shape[0] > 0:
            self._food_coords = self.food_sources[:, :2].astype(np.int32, copy=True)
        else:
            self._food_coords = np.empty((0,2), dtype=np.int32)

        if self.doors:
            self._door_coords = np.array([(d.y, d.x) for d in self.doors], dtype=np.int32)
        else:
            self._door_coords = np.empty((0,2), dtype=np.int32)

        if self.buttons:
            self._button_coords = np.array([(b.y, b.x) for b in self.buttons], dtype=np.int32)
        else:
            self._button_coords = np.empty((0,2), dtype=np.int32)

        self._update_passable_cache()

    def _update_passable_cache(self):
        if self.static_grid is None:
            return
        not_obstacle = (self.static_grid != TileType.OBSTACLE)
        door_open = (self.door_open_array == 1)
        door_closed = (self.static_grid == TileType.DOOR_CLOSED)
        passable = not_obstacle & (~door_closed | door_open)
        self._passable_cache = passable

    def _is_passable(self, y: int, x: int) -> bool:
        return self._passable_cache[y, x]

    def _can_move_to(self, y: int, x: int) -> bool:
        return (0 <= y < self.grid_size and 0 <= x < self.grid_size and self._passable_cache[y, x])

    # ------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[0, :] = TileType.OBSTACLE
        self.grid[-1, :] = TileType.OBSTACLE
        self.grid[:, 0] = TileType.OBSTACLE
        self.grid[:, -1] = TileType.OBSTACLE
        self.grid = add_obstacles_connectivity(self.grid, self.n_obstacles)

        self.food_positions_cache = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.door_open_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.button_broken_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self._init_food_sources()
        self._init_doors_and_buttons()
        self._cache_reset_state()

        spawn_idx = np.random.randint(self._spawn_cells.shape[0])
        self.agent_y = int(self._spawn_cells[spawn_idx, 0])
        self.agent_x = int(self._spawn_cells[spawn_idx, 1])

        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        self.last_action = ENV_ACTIONS_START

        self._n_doors_active = len(self.doors)
        self._n_buttons_working = len(self.buttons)

        info = self._info
        info["energy"] = self.energy
        info["steps"] = self.steps
        info["position"] = np.array([self.agent_y, self.agent_x], dtype=np.int32)
        info["task_class"] = self.task_class
        info["complexity_level"] = self.complexity_level
        info["n_doors"] = len(self.doors)
        info["n_buttons"] = len(self.buttons)

        return self._get_observation(), info

    # ------------------------------------------------------------
    # soft_reset()
    # ------------------------------------------------------------
    def soft_reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        spawn_idx = np.random.randint(self._spawn_cells.shape[0])
        self.agent_y = int(self._spawn_cells[spawn_idx, 0])
        self.agent_x = int(self._spawn_cells[spawn_idx, 1])

        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        self.last_action = ENV_ACTIONS_START

        if self.food_sources is not None and self.food_sources.shape[0] > 0:
            n = self.food_sources.shape[0]
            self._regen_buffer[:n] = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME, size=n)
            self.food_sources[:, FOOD_INTERVAL_INDEX] = self._regen_buffer[:n]
            self.food_sources[:, FOOD_EXISTS_INDEX] = 1
            self.food_sources[:, FOOD_COLLECTION_COUNT_INDEX] = 0
            if self._food_coords.shape[0] > 0:
                ys = self._food_coords[:, 0]
                xs = self._food_coords[:, 1]
                self.food_positions_cache[ys, xs] = 1

        if self._door_coords.shape[0] > 0:
            self.door_open_array[self._door_coords[:, 0], self._door_coords[:, 1]] = 0
        for door in self.doors:
            door.is_open = False
            door.timer = 0
            door.can_be_opened = True

        if self._button_coords.shape[0] > 0:
            self.button_broken_array[self._button_coords[:, 0], self._button_coords[:, 1]] = 0
        for button in self.buttons:
            button.is_broken = False

        self._update_passable_cache()

        self._n_doors_active = len(self.doors)
        self._n_buttons_working = len(self.buttons)

        info = self._info
        info["energy"] = self.energy
        info["steps"] = self.steps
        info["position"] = np.array([self.agent_y, self.agent_x], dtype=np.int32)
        info["task_class"] = self.task_class
        info["complexity_level"] = self.complexity_level
        info["n_doors"] = len(self.doors)
        info["n_buttons"] = len(self.buttons)

        return self._get_observation(), info

    # ------------------------------------------------------------
    # Step, door updates, button press
    # ------------------------------------------------------------
    def _update_door_states(self):
        changed = False
        for door in self.doors:
            old_open = door.is_open
            door.update((self.agent_y, self.agent_x))
            if door.is_open != old_open:
                self.door_open_array[door.y, door.x] = 1 if door.is_open else 0
                changed = True
        if changed:
            self._update_passable_cache()

    def _check_button_press(self, button_y: int, button_x: int) -> bool:
        for button in self.buttons:
            if button.y == button_y and button.x == button_x:
                if button.is_broken:
                    return False
                success = button.press()
                if button.is_broken:
                    self.button_broken_array[button_y, button_x] = 1
                    self._n_buttons_working -= 1
                    door = self.doors[button.door_idx]
                    other_buttons_working = any(b for b in self.buttons if b.door_idx == button.door_idx and not b.is_broken)
                    if not other_buttons_working:
                        door.can_be_opened = False
                        self._n_doors_active -= 1
                    return False
                if success and 0 <= button.door_idx < len(self.doors):
                    door = self.doors[button.door_idx]
                    if door.open():
                        self.door_open_array[door.y, door.x] = 1
                        self._update_passable_cache()
                        return True
                break
        return False

    def _get_adjacent_button_positions(self, y: int, x: int) -> List[Tuple[int, int]]:
        adjacent = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.static_grid[ny, nx] == TileType.BUTTON:
                    adjacent.append((ny, nx))
        return adjacent

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            return self._get_observation(), 0.0, True, True, {}

        self._update_door_states()

        button_pressed = False
        moved = False
        y, x = self.agent_y, self.agent_x

        if action == Actions.BUTTON:
            for by, bx in self._get_adjacent_button_positions(y, x):
                if self._check_button_press(by, bx):
                    button_pressed = True
                    break
        else:
            moved = True
            if action == Actions.LEFT:
                if x > 0 and self._can_move_to(y, x-1):
                    x -= 1
            elif action == Actions.RIGHT:
                if x < self.grid_size-1 and self._can_move_to(y, x+1):
                    x += 1
            elif action == Actions.UP:
                if y > 0 and self._can_move_to(y-1, x):
                    y -= 1
            elif action == Actions.DOWN:
                if y < self.grid_size-1 and self._can_move_to(y+1, x):
                    y += 1

        if moved:
            self.agent_y, self.agent_x = y, x

        energy_gained = 0.0
        if moved:
            # Use the regrown buffer – allocate once (size = food_sources.shape[0])
            if not hasattr(self, '_regrown_buffer') or self._regrown_buffer.size < self.food_sources.shape[0]:
                self._regrown_buffer = np.zeros(self.food_sources.shape[0], dtype=np.int32)
            energy_gained, regrown_cnt = food_step(
                y, x, self.food_sources, self.food_energy, self._regrown_buffer
            )
            if energy_gained > 0:
                self.food_positions_cache[y, x] = 0
            for idx in range(regrown_cnt):
                i = self._regrown_buffer[idx]
                yf = self.food_sources[i, 0]
                xf = self.food_sources[i, 1]
                self.food_positions_cache[yf, xf] = 1

        self.energy = (self.energy * self.energy_decay + energy_gained - self.energy_per_step)
        self.energy = max(0.0, min(self.energy, 100.0))

        self.steps += 1
        self.last_action = action
        terminated = (self.steps >= self.max_steps or self.energy <= 0)
        truncated = False
        self.done = terminated or truncated

        reward = 0.01
        if energy_gained > 0:
            reward += 1.0
        if action == Actions.BUTTON:
            reward += 0.5 if button_pressed else -0.1
        if self.energy < 10:
            reward -= 0.1

        obs = self._get_observation()
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': np.array([self.agent_y, self.agent_x], dtype=np.int32),
            'food_collected': energy_gained > 0,
            'button_pressed': button_pressed,
            'action_taken': action,
            'task_class': self.task_class,
            'complexity_level': self.complexity_level,
            'n_doors_active': self._n_doors_active,
            'n_buttons_working': self._n_buttons_working
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        return get_observation_optimized(
            self.agent_y, self.agent_x,
            self.static_grid, self.last_action, self.energy,
            self.food_positions_cache, self.door_open_array, self.button_broken_array
        )

    # ------------------------------------------------------------
    # Render (unchanged)
    # ------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_render_buffer') or self._render_buffer is None:
            cell_size = max(1, self.render_size // self.grid_size)
            self._render_buffer = np.zeros((self.grid_size * cell_size, self.grid_size * cell_size, 3), dtype=np.uint8)
            self._cell_size = cell_size

        self._render_buffer.fill(0)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                tile = self.static_grid[y, x]
                if tile == TileType.DOOR_CLOSED and self.door_open_array[y, x] == 1:
                    tile = TileType.DOOR_OPEN
                elif tile == TileType.BUTTON and self.button_broken_array[y, x] == 1:
                    tile = TileType.BUTTON_BROKEN
                color = self.colors[tile]
                y0, y1 = y * self._cell_size, (y+1) * self._cell_size
                x0, x1 = x * self._cell_size, (x+1) * self._cell_size
                self._render_buffer[y0:y1, x0:x1] = color

        for door in self.doors:
            cy = int((door.y + 0.5) * self._cell_size)
            cx = int((door.x + 0.5) * self._cell_size)
            fs = self._cell_size / 30.0
            thick = max(1, int(self._cell_size / 20))
            r = max(2, self._cell_size // 4)
            color = (50,50,50) if door.is_open else (200,200,200)
            cv2.circle(self._render_buffer, (cx, cy), r, color, -1)
            text = str(door.door_number)
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0]
            cv2.putText(self._render_buffer, text, (cx - tw//2, cy + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), thick)

        for button in self.buttons:
            cy = int((button.y + 0.5) * self._cell_size)
            cx = int((button.x + 0.5) * self._cell_size)
            fs = self._cell_size / 30.0
            thick = max(1, int(self._cell_size / 20))
            r = max(2, self._cell_size // 5)
            color = (200,0,0) if button.is_broken else (0,0,200)
            cv2.circle(self._render_buffer, (cx, cy), r, color, -1)
            door = self.doors[button.door_idx]
            text = str(door.door_number)
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0]
            cv2.putText(self._render_buffer, text, (cx - tw//2, cy + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), thick)

        if self.food_sources is not None:
            for i in range(self.food_sources.shape[0]):
                y, x, delay, has_food, _ = self.food_sources[i]
                cy = int((y + 0.5) * self._cell_size)
                cx = int((x + 0.5) * self._cell_size)
                if has_food:
                    r = max(1, self._cell_size // 3)
                    cv2.circle(self._render_buffer, (cx, cy), r, (0,255,0), -1)
                else:
                    r = max(1, self._cell_size // 5)
                    cv2.circle(self._render_buffer, (cx, cy), r, (0,0,0), -1)
                    fs = self._cell_size / 40.0
                    thick = max(1, int(self._cell_size / 30))
                    text = str(delay)
                    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0]
                    cv2.putText(self._render_buffer, text, (cx - tw//2, cy + th//2),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), thick)

        ay, ax = self.agent_y, self.agent_x
        cy = int((ay + 0.5) * self._cell_size)
        cx = int((ax + 0.5) * self._cell_size)
        r = max(1, self._cell_size // 2)
        cv2.circle(self._render_buffer, (cx, cy), r, (255,255,255), -1)

        info_line = f"Energy: {self.energy:.1f} | Step: {self.steps}/{self.max_steps}"
        info_line += f" | Task: {self.task_class} (Lvl: {self.complexity_level:.1f})"
        doors_line = f"Doors: {len(self.doors)} | Buttons: {len(self.buttons)}"
        cv2.putText(self._render_buffer, info_line, (10,15), cv2.QT_FONT_NORMAL, 0.55, (255,255,255), 1)
        cv2.putText(self._render_buffer, doors_line, (10,35), cv2.QT_FONT_NORMAL, 0.55, (255,255,255), 1)

        return self._render_buffer


class VectorGridMazeWorld(GridMazeWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._render_buffer = None

    def render(self):
        return None

    def close(self):
        pass