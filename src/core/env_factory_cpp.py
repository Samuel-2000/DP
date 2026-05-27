# src/core/env_factory_cpp.py
"""
Environment factory for C++ maze_core module

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

from typing import Dict, Any
import maze_core
from src.core.constants import DEFAULT_RENDER_SIZE

class EnvironmentFactoryCPP:
    """Factory for creating C++ maze environments"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any], test_mode: bool = False):
        """Create a single C++ environment instance from configuration dictionary"""
        if 'environment' in config:
            env_config = config['environment'].copy()
        else:
            env_config = config.copy()
        
        # Set render size based on mode
        if test_mode:
            env_config['render_size'] = DEFAULT_RENDER_SIZE
        else:
            env_config['render_size'] = 0
        
        # Convert None to appropriate defaults for C++ constructor
        n_doors_val = env_config.get('n_doors')
        if n_doors_val is None:
            n_doors_val = 0
        
        n_buttons_val = env_config.get('n_buttons_per_door')
        if n_buttons_val is None:
            n_buttons_val = 0
        
        break_prob_val = env_config.get('button_break_probability')
        if break_prob_val is None:
            break_prob_val = 0.0
        
        # Create C++ environment
        return maze_core.GridMazeWorld(
            grid_size=env_config.get('grid_size', 11),
            max_steps=env_config.get('max_steps', 100),
            n_food_sources=env_config.get('n_food_sources', 0),
            food_energy=env_config.get('food_energy', 10.0),
            initial_energy=env_config.get('initial_energy', 30.0),
            energy_decay=env_config.get('energy_decay', 0.98),
            energy_per_step=env_config.get('energy_per_step', 0.1),
            task_class=env_config.get('task_class', 'basic'),
            complexity_level=env_config.get('complexity_level', 0.5),
            n_doors=n_doors_val,
            door_open_duration=env_config.get('door_open_duration', 10),
            door_close_duration=env_config.get('door_close_duration', 20),
            n_buttons_per_door=n_buttons_val,
            button_break_probability=break_prob_val
        )
    
    @staticmethod
    def create_vectorized(num_envs: int, config: Dict[str, Any], base_seed: int):
        """Create a vectorized C++ environment"""
        if 'environment' in config:
            env_config = config['environment'].copy()
        else:
            env_config = config.copy()
        
        env_config['render_size'] = 0  # Disable rendering for vectorized
        
        # Convert None to appropriate defaults
        n_doors_val = env_config.get('n_doors')
        if n_doors_val is None:
            n_doors_val = -1
        
        n_buttons_val = env_config.get('n_buttons_per_door')
        if n_buttons_val is None:
            n_buttons_val = -1
        
        break_prob_val = env_config.get('button_break_probability')
        if break_prob_val is None:
            break_prob_val = -1.0
        
        return maze_core.VectorizedMazeEnv(
            num_envs=num_envs,
            grid_size=env_config.get('grid_size', 11),
            max_steps=env_config.get('max_steps', 100),
            n_food_sources=env_config.get('n_food_sources', 0),
            food_energy=env_config.get('food_energy', 10.0),
            initial_energy=env_config.get('initial_energy', 30.0),
            energy_decay=env_config.get('energy_decay', 0.98),
            energy_per_step=env_config.get('energy_per_step', 0.1),
            task_class=env_config.get('task_class', 'basic'),
            complexity_level=env_config.get('complexity_level', 0.5),
            n_doors=n_doors_val,
            door_open_duration=env_config.get('door_open_duration', 10),
            door_close_duration=env_config.get('door_close_duration', 20),
            n_buttons_per_door=n_buttons_val,
            button_break_probability=break_prob_val,
            base_seed=base_seed
        )