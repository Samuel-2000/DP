#!/usr/bin/env python3
"""
test_environment.py - Create a single environment with fixed seed,
render it, and save a screenshot using the C++ module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# First, ensure C++ module is built and available
from src.core.cpp_build import ensure_cpp_module
ensure_cpp_module()

import numpy as np
import cv2
import maze_core
from src.core.utils import seed_everything

# Parameters
config = {
    'grid_size': 51,
    'max_steps': 100,
    'n_food_sources': 0,
    'food_energy': 10.0,
    'initial_energy': 30.0,
    'energy_decay': 0.98,
    'energy_per_step': 0.1,
    'task_class': 'complex',
    'complexity_level': 0.5,
    'n_doors': 0,
    'door_open_duration': 10,
    'door_close_duration': 20,
    'n_buttons_per_door': 0,
    'button_break_probability': 0.0
}

SEED = 42
seed_everything(SEED)

env = maze_core.GridMazeWorld(
    grid_size=config['grid_size'],
    max_steps=config['max_steps'],
    n_food_sources=config['n_food_sources'],
    food_energy=config['food_energy'],
    initial_energy=config['initial_energy'],
    energy_decay=config['energy_decay'],
    energy_per_step=config['energy_per_step'],
    task_class=config['task_class'],
    complexity_level=config['complexity_level'],
    n_doors=config['n_doors'],
    door_open_duration=config['door_open_duration'],
    door_close_duration=config['door_close_duration'],
    n_buttons_per_door=config['n_buttons_per_door'],
    button_break_probability=config['button_break_probability']
)

obs, info = env.reset(seed=SEED)
print("Environment reset with seed", SEED)

frame = env.render(render_size=512)
if frame is not None:
    output_path = Path("test_environment_screenshot.png")
    cv2.imwrite(str(output_path), frame)
    print(f"Screenshot saved to {output_path}")
else:
    print("Render returned None")