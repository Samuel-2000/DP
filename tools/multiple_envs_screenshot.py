#!/usr/bin/env python3
"""
test_environment_mosaic_pdf.py
Vector‑layout PDF (400 DPI) with 2× super‑sampled maze images.
No title, vertical row labels perfectly centred on the image rows.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.cpp_build import ensure_cpp_module
ensure_cpp_module()

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import maze_core
from src.core.utils import seed_everything

# ----------------------------------------------------------------------
# Layout (inches) – tuned for height ≈ 1.4 × width, DPI = 400
# ----------------------------------------------------------------------
DPI           = 400
PANEL_SIZE_IN = 3.0                     # 1200 px ÷ 400 dpi
GAP_IN        = 0.15
MARGIN_LEFT   = 1.5                     # comfortable space for vertical labels
MARGIN_RIGHT  = 0.4
MARGIN_TOP    = 0.67                    # room for column labels
MARGIN_BOTTOM = 0.67

GRID_SIZES = [11, 19, 29, 59]
ROW_COMBOS = [
    ("basic",   0.0),
    ("basic",   0.5),
    ("basic",   1.0),
    ("complex", 0.0),
    ("complex", 0.5),
    ("complex", 1.0),
]
N_COLS = len(GRID_SIZES)
N_ROWS = len(ROW_COMBOS)

page_width_in  = MARGIN_LEFT + N_COLS * PANEL_SIZE_IN + (N_COLS - 1) * GAP_IN + MARGIN_RIGHT
page_height_in = MARGIN_TOP  + N_ROWS * PANEL_SIZE_IN + (N_ROWS - 1) * GAP_IN + MARGIN_BOTTOM

assert abs(page_height_in / page_width_in - 1.4) < 0.05, (
    f"Aspect ratio {page_height_in / page_width_in:.2f} ≠ 1.4, adjust margins"
)

# ----------------------------------------------------------------------
# Environment parameters
# ----------------------------------------------------------------------
FIXED_PARAMS = dict(
    max_steps=100,
    n_food_sources=0,
    food_energy=10.0,
    initial_energy=30.0,
    energy_decay=0.98,
    energy_per_step=0.1,
    n_doors=0,
    n_buttons_per_door=0,
    door_open_duration=10,
    door_close_duration=20,
    button_break_probability=0.0,
)

BASE_SEED = 43

# ----------------------------------------------------------------------
def render_env_super_sampled(grid_size, task_class, complexity_level, seed, display_px):
    """Render at 2× display resolution, then downsample for anti‑aliasing."""
    seed_everything(seed)

    env = maze_core.GridMazeWorld(
        grid_size=grid_size,
        task_class=task_class,
        complexity_level=complexity_level,
        **FIXED_PARAMS,
    )
    obs, info = env.reset(seed=seed)

    # 2× super‑sampling
    render_size = 2 * display_px
    cell_px = max(1, render_size // grid_size)
    internal_size = grid_size * cell_px
    frame_bgr = env.render(render_size=internal_size)
    if frame_bgr is None:
        frame_bgr = np.zeros((display_px, display_px, 3), dtype=np.uint8)
    else:
        frame_bgr = cv2.resize(frame_bgr, (display_px, display_px),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------------------------
# Build the PDF
# ----------------------------------------------------------------------
print(f"Creating PDF (400 DPI, page {page_width_in:.1f}\"×{page_height_in:.1f}\") ...")
fig = plt.figure(figsize=(page_width_in, page_height_in), dpi=DPI)

# Column labels (grid sizes), placed above the top row
for col_idx, gs in enumerate(GRID_SIZES):
    x_center = MARGIN_LEFT + col_idx * (PANEL_SIZE_IN + GAP_IN) + PANEL_SIZE_IN / 2
    y_label = MARGIN_TOP - 0.4
    fig.text(x_center / page_width_in, y_label / page_height_in,
             f"Grid {gs}×{gs}", ha='center', va='center', fontsize=10)

# Vertical row labels (task + complexity), centred on the row
# Correct order: row 0 is at the TOP of the page
for row_idx, (task, comp) in enumerate(ROW_COMBOS):
    # y_center measured from TOP of page in inches
    y_center_top = MARGIN_TOP + row_idx * (PANEL_SIZE_IN + GAP_IN) + PANEL_SIZE_IN / 2
    # Convert to matplotlib's coordinate (measured from BOTTOM)
    y_center_mpl = page_height_in - y_center_top
    x_label = MARGIN_LEFT - 0.3
    label_str = f"{task}, C={comp}"
    fig.text(x_label / page_width_in, y_center_mpl / page_height_in,
             label_str, rotation=90, ha='center', va='center', fontsize=10)

# Place each super‑sampled maze image
display_px = int(PANEL_SIZE_IN * DPI)   # 1200 px at 400 DPI

for row_idx, (task, comp) in enumerate(ROW_COMBOS):
    for col_idx, gs in enumerate(GRID_SIZES):
        env_seed = BASE_SEED + row_idx * N_COLS + col_idx
        print(f"  Rendering: grid={gs}, task={task}, complexity={comp}, seed={env_seed}")

        rgb_img = render_env_super_sampled(gs, task, comp, env_seed, display_px)

        # Position in figure coordinates (matplotlib uses bottom-left origin)
        # bottom measured from BOTTOM of page
        bottom = (page_height_in - MARGIN_TOP - (row_idx + 1) * PANEL_SIZE_IN - row_idx * GAP_IN) / page_height_in
        left   = (MARGIN_LEFT + col_idx * (PANEL_SIZE_IN + GAP_IN)) / page_width_in
        width  = PANEL_SIZE_IN / page_width_in
        height = PANEL_SIZE_IN / page_height_in

        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(rgb_img, interpolation='bilinear')
        ax.axis('off')

output_path = Path("test_environment_mosaic.pdf")
fig.savefig(str(output_path), dpi=DPI, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(f"High‑quality PDF saved to {output_path}")