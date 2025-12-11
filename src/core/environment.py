"""
Grid Maze Environment using Gymnasium
Combines best optimizations from GridMazeWorld
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from numba import jit, njit, prange
from typing import Tuple, Dict, Any, Optional
import enum


class TileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    FOOD_SOURCE = 2
    FOOD = 3
    AGENT = 4


class Actions(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    START = 5


@njit(cache=True, parallel=True)
def add_obstacles_connectivity(grid: np.ndarray, n_obstacles: int) -> np.ndarray:
    """Add obstacles while maintaining connectivity - OPTIMIZED VERSION"""
    h, w = grid.shape
    total_cells = h * w
    
    # Get empty cells using vectorized operations
    empty_mask = (grid == 0).flatten()
    empty_ids = np.where(empty_mask)[0]
    n_empty = len(empty_ids)
    
    if n_obstacles > n_empty - 1:
        n_obstacles = n_empty - 1
    
    # Pre-allocate BFS arrays
    visited = np.zeros(total_cells, dtype=np.uint8)
    queue = np.empty(total_cells, dtype=np.int32)
    
    added = 0
    for _ in range(n_obstacles):
        for attempt in range(n_empty):
            pick = np.random.randint(0, n_empty)
            cell = empty_ids[pick]
            r, c = cell // w, cell % w
            
            # Try placing obstacle
            grid[r, c] = 1
            
            # Find start for BFS (first empty cell except current)
            start = -1
            for j in range(n_empty):
                if j == pick:
                    continue
                nid = empty_ids[j]
                rr, cc = nid // w, nid % w
                if grid[rr, cc] == 0:
                    start = nid
                    break
            
            if start < 0:
                grid[r, c] = 0
                continue
            
            # BFS with optimizations
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
                
                # Check neighbors using precomputed indices
                if cr > 0 and grid[cr-1, cc] == 0:
                    nid = (cr-1) * w + cc
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                
                if cr < h-1 and grid[cr+1, cc] == 0:
                    nid = (cr+1) * w + cc
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                
                if cc > 0 and grid[cr, cc-1] == 0:
                    nid = cr * w + (cc-1)
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                
                if cc < w-1 and grid[cr, cc+1] == 0:
                    nid = cr * w + (cc+1)
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
            
            if reach == n_empty - 1:
                # Success - update empty cells
                empty_ids[pick] = empty_ids[n_empty - 1]
                n_empty -= 1
                added += 1
                break
            else:
                grid[r, c] = 0
    
    return grid


@njit(cache=True)
def food_step(agent_y: int, agent_x: int, 
                   food_sources: np.ndarray, 
                   food_energy: float) -> float:
    """food processing with minimal branching"""
    energy_gained = 0.0
    n_food = food_sources.shape[0]
    
    for i in prange(n_food):
        y, x, time_left, has_food = food_sources[i]
        
        # Check if agent is on this food source
        if agent_y == y and agent_x == x and has_food:
            energy_gained += food_energy
            food_sources[i, 2] = np.random.randint(5, 15)  # Regeneration time
            food_sources[i, 3] = 0  # Mark as empty
        elif time_left > 0: # Update regeneration timer
            food_sources[i, 2] = time_left - 1
        elif time_left == 0:
            food_sources[i, 3] = 1  # Regenerate food
    
    return energy_gained


@njit(cache=True)
def get_observation(y: int, x: int, 
                             grid: np.ndarray, 
                             food_sources: np.ndarray,
                             last_action: int, 
                             energy: float,
                             food_positions_cache: np.ndarray) -> np.ndarray:
    """JIT-compiled observation generation"""
    obs = np.empty(10, dtype=np.int32)
    
    # Neighbor offsets in order: NW, N, NE, W, E, SW, S, SE
    offsets = np.array([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1], [0, 1],
                        [1, -1], [1, 0], [1, 1]], dtype=np.int32)
    
    grid_h, grid_w = grid.shape
    
    for i in range(8):
        ny = y + offsets[i, 0]
        nx = x + offsets[i, 1]
        
        if 0 <= ny < grid_h and 0 <= nx < grid_w:
            # Check if position has food using cache
            if food_positions_cache[ny, nx] > 0:
                obs[i] = TileType.FOOD.value
            else:
                obs[i] = grid[ny, nx]
        else:
            obs[i] = TileType.OBSTACLE.value
    
    # Last action (6-11)
    obs[8] = last_action + 6
    
    # Energy level (12-19)
    energy_scaled = int((energy / 100.0) * 7) + 12
    if energy_scaled < 12:
        energy_scaled = 12
    elif energy_scaled > 19:
        energy_scaled = 19
    obs[9] = energy_scaled
    
    return obs


class GridMazeWorld(gym.Env):
    """Grid Maze Environment"""
    
    def __init__(self, 
                 grid_size: int = 11,
                 max_steps: int = 100,
                 obstacle_fraction: float = 0.25,
                 n_food_sources: int = 4,
                 food_energy: float = 10.0,
                 initial_energy: float = 30.0,
                 energy_decay: float = 0.98,
                 energy_per_step: float = 0.1,
                 render_size: int = 512):
        
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_food_sources = n_food_sources
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size
        
        # Calculate obstacle count
        self.n_obstacles = int((grid_size - 2) ** 2 * obstacle_fraction)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=0, high=19, shape=(10,), dtype=np.int32
        )
        
        # Initialize state with pre-allocated arrays
        self.grid = None
        self.food_sources = None
        self.food_positions_cache = None  # 2D array for fast food lookup
        self.agent_pos = None
        self.energy = None
        self.steps = None
        self.done = None
        self.last_action = None
        
        # Pre-compute neighbor offsets
        self.neighbor_offsets = np.array([[-1, -1], [-1, 0], [-1, 1],
                                         [0, -1], [0, 1],
                                         [1, -1], [1, 0], [1, 1]], dtype=np.int32)
        
        # Colors for rendering (only if needed)
        self.colors = {
            TileType.EMPTY.value: (40, 40, 40),
            TileType.OBSTACLE.value: (100, 100, 100),
            TileType.FOOD_SOURCE.value: (200, 50, 50),
            TileType.FOOD.value: (50, 200, 50),
            TileType.AGENT.value: (50, 50, 200)
        }
        
        # Pre-allocated observation buffer to avoid allocations
        self._obs_buffer = np.zeros(10, dtype=np.int32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state - OPTIMIZED"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create grid with borders using vectorized operations
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[0, :] = TileType.OBSTACLE.value
        self.grid[-1, :] = TileType.OBSTACLE.value
        self.grid[:, 0] = TileType.OBSTACLE.value
        self.grid[:, -1] = TileType.OBSTACLE.value
        
        # Add obstacles
        self.grid = add_obstacles_connectivity(self.grid, self.n_obstacles)
        
        # Initialize food sources
        self._init_food_sources()
        
        # Initialize food positions cache
        self.food_positions_cache = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._update_food_cache()
        
        # Place agent using vectorized operations
        empty_cells = np.argwhere(self.grid == TileType.EMPTY.value)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]
        
        # Reset state variables
        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        self.last_action = Actions.START.value
        
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': self.agent_pos.copy()
        }
        
        # Get observation using fast JIT function
        obs = get_observation(
            self.agent_pos[0], self.agent_pos[1],
            self.grid, self.food_sources,
            self.last_action, self.energy,
            self.food_positions_cache
        )
        
        return obs, info
    
    def _init_food_sources(self):
        """food source initialization"""
        empty_cells = np.argwhere(self.grid == TileType.EMPTY.value)
        indices = np.random.choice(len(empty_cells), self.n_food_sources, replace=False)
        
        self.food_sources = np.zeros((self.n_food_sources, 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(5, 15)
            self.food_sources[i] = [y, x, regen_time, 1]  # Start with food
    
    def _update_food_cache(self):
        """Update the food positions cache"""
        self.food_positions_cache.fill(0)
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food = self.food_sources[i]
            if has_food:
                self.food_positions_cache[y, x] = 1
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """step function with minimal allocations"""
        if self.done:
            obs = get_observation(
                self.agent_pos[0], self.agent_pos[1],
                self.grid, self.food_sources,
                self.last_action, self.energy,
                self.food_positions_cache
            )
            return obs, 0.0, True, True, {}
        
        # Move agent with bounds checking
        y, x = self.agent_pos
        
        if action == Actions.LEFT.value:
            if x > 0 and self.grid[y, x-1] == 0:
                x -= 1
        elif action == Actions.RIGHT.value:
            if x < self.grid_size-1 and self.grid[y, x+1] == 0:
                x += 1
        elif action == Actions.UP.value:
            if y > 0 and self.grid[y-1, x] == 0:
                y -= 1
        elif action == Actions.DOWN.value:
            if y < self.grid_size-1 and self.grid[y+1, x] == 0:
                y += 1
        
        self.agent_pos = np.array([y, x])
        
        # Process food with JIT-compiled function
        energy_gained = food_step(y, x, self.food_sources, self.food_energy)
        
        # Update food cache if food was collected
        if energy_gained > 0:
            self.food_positions_cache[y, x] = 0
        
        # Update energy with single calculation
        self.energy = (self.energy * self.energy_decay + 
                      energy_gained - self.energy_per_step)
        
        # Clip energy
        self.energy = max(0.0, min(self.energy, 100.0))
        
        # Update state
        self.steps += 1
        self.last_action = action
        
        # Check termination
        terminated = (self.steps >= self.max_steps or self.energy <= 0)
        truncated = False
        self.done = terminated or truncated
        
        # Calculate reward - optimized to avoid branches
        reward = 0.01
        if energy_gained > 0:
            reward += 1.0
        if self.energy < 10:
            reward -= 0.1
        
        # Update food cache if food regenerated
        if self.steps % 2 == 0:  # Only update cache every other step
            self._update_food_cache()
        
        # Get observation
        obs = get_observation(
            y, x, self.grid, self.food_sources,
            action, self.energy,
            self.food_positions_cache
        )
        
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': self.agent_pos.copy(),
            'food_collected': energy_gained > 0
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state - only called for visualization"""
        if not hasattr(self, '_render_buffer') or self._render_buffer is None:
            cell_size = self.render_size // self.grid_size
            self._render_buffer = np.zeros(
                (self.grid_size * cell_size, self.grid_size * cell_size, 3), 
                dtype=np.uint8
            )
            self._cell_size = cell_size
        
        # Clear buffer
        self._render_buffer.fill(0)
        
        # Draw grid cells
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.colors[self.grid[y, x]]
                y_start = y * self._cell_size
                x_start = x * self._cell_size
                self._render_buffer[y_start:y_start+self._cell_size, 
                                   x_start:x_start+self._cell_size] = color
        
        # Draw food
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food = self.food_sources[i]
            if has_food:
                center_y = int((y + 0.5) * self._cell_size)
                center_x = int((x + 0.5) * self._cell_size)
                radius = self._cell_size // 3
                cv2.circle(self._render_buffer, (center_x, center_y), 
                          radius, (0, 255, 0), -1)
        
        # Draw agent
        ay, ax = self.agent_pos
        center_y = int((ay + 0.5) * self._cell_size)
        center_x = int((ax + 0.5) * self._cell_size)
        radius = self._cell_size // 2
        cv2.circle(self._render_buffer, (center_x, center_y), 
                  radius, (255, 255, 255), -1)
        
        # Add info overlay
        info = f"Energy: {self.energy:.1f} | Step: {self.steps}/{self.max_steps}"
        cv2.putText(self._render_buffer, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return self._render_buffer
    
    def close(self):
        """Close environment"""
        if hasattr(self, '_render_buffer'):
            self._render_buffer = None
        cv2.destroyAllWindows()


# vectorized environment alternative
class VectorGridMazeWorld(GridMazeWorld):
    """version optimized for vectorized use (no rendering overhead)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Disable render components for pure training
        self._render_buffer = None
    
    def render(self):
        """Override to return None when not needed"""
        return None
    
    def close(self):
        """Minimal close"""
        pass