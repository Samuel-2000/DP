#!/usr/bin/env python3
"""
test_environment.py - Create a single environment with fixed seed,
render it, and save a screenshot to verify the incremental mask changes didn't break anything.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from src.core.environment import GridMazeWorld

# Parameters (same as your typical training)
config = {
    'grid_size': 11,
    'max_steps': 100,
    'n_food_sources': 4,
    'food_energy': 10.0,
    'initial_energy': 30.0,
    'energy_decay': 0.98,
    'energy_per_step': 0.1,
    'render_size': 512,        # large for screenshot
    'task_class': 'doors',     # can be 'basic', 'doors', 'buttons', 'complex'
    'complexity_level': 1.0,
    'n_doors': 5,
    'door_open_duration': 10,
    'door_close_duration': 20,
    'n_buttons_per_door': 4,
    'button_break_probability': 0.0
}

# Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Create environment
env = GridMazeWorld(**config)

# Reset with seed
obs, info = env.reset(seed=SEED)
print("Environment reset with seed", SEED)
print(f"Doors: {len(env.doors)}, Buttons: {len(env.buttons)}")
print(f"Agent position: {env.agent_pos}")

# Render
frame = env.render()
if frame is not None:
    # Save screenshot
    output_path = Path("test_environment_screenshot.png")
    cv2.imwrite(str(output_path), frame)
    print(f"Screenshot saved to {output_path}")
else:
    print("Render returned None")

# Optional: display using OpenCV (uncomment to see live)
# cv2.imshow("Test Environment", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()