"""
Visualizer: handles fog of war, agent view, trail, and video saving.
Supports interactive toggling during testing.
Frame storage is now done manually by the agent to allow overlays.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Set, List, Optional
import imageio


class Visualizer:
    """Handles all visualisation effects and video saving for an episode."""

    def __init__(self, env, save_video: bool, video_path: Optional[Path],
                 agent_view: bool, fog_of_war: bool, show_trail: bool, as_gif: bool, render_size: int):
        self.env = env
        self.save_video = save_video
        self.agent_view = agent_view
        self.fog_of_war = fog_of_war
        self.show_trail = show_trail
        self.as_gif = as_gif

        # Ensure render_size is a multiple of 16 (required for MP4 encoding)
        if save_video:
            self.render_size = self._round_to_multiple_of_16(render_size)
        else:
            self.render_size = render_size

        self.visited: Set[Tuple[int, int]] = set()
        self.trail: List[Tuple[int, int, int]] = []  # (y, x, step)
        self.frames = []  # filled manually by the agent

        # Precompute the height of the UI text area (as drawn by the C++ environment)
        self._text_height = self._get_text_height()

        # Prepare output path if saving is requested
        if save_video and video_path:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_path = video_path
        else:
            self.video_path = None

    # ------------------------------------------------------------------
    #  Interactive toggles
    # ------------------------------------------------------------------
    def toggle_fog_of_war(self):
        self.fog_of_war = not self.fog_of_war
        print(f"Fog of war: {'ON' if self.fog_of_war else 'OFF'}")
        if not self.fog_of_war:
            self.visited.clear()

    def toggle_agent_view(self):
        self.agent_view = not self.agent_view
        print(f"Agent view (crop): {'ON' if self.agent_view else 'OFF'}")

    def toggle_show_trail(self):
        self.show_trail = not self.show_trail
        print(f"Trail: {'ON' if self.show_trail else 'OFF'}")

    # ------------------------------------------------------------------
    #  Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _round_to_multiple_of_16(size: int) -> int:
        return ((size + 15) // 16) * 16

    def _get_text_height(self) -> int:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line1 = "Energy: 0.0"
        line2 = "Step: 0/0"
        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        return h1 + h2 + 6

    def reset(self):
        self.visited.clear()
        self.trail.clear()
        self.frames.clear()

    def render(self, step: int, show_text: bool = True) -> np.ndarray:
        """
        Get the environment frame, apply visual effects, and return it.
        show_text: if False, suppress door/button numbers and food timers.
        This method does NOT store the frame for video – that is done manually by the agent.
        """
        raw_frame = self.env.render(self.render_size, show_text)
        if raw_frame is None:
            return None

        frame = raw_frame.copy()
        y, x = self.env.agent_y, self.env.agent_x
        grid_size = self.env.grid_size
        cell_size = frame.shape[0] // grid_size

        # Update state for fog of war and trail
        if self.fog_of_war or self.show_trail:
            self.visited.add((int(y), int(x)))
        if self.show_trail:
            self.trail.append((int(y), int(x), step))

        # Expand visited neighbourhood for fog of war (3x3 around agent)
        if self.fog_of_war:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = int(y) + dy, int(x) + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        self.visited.add((ny, nx))

        # Apply fog of war
        if self.fog_of_war:
            frame = self._apply_fog_of_war(frame, grid_size, cell_size)

        # Apply agent view (crop)
        crop_offset = (0, 0)
        if self.agent_view:
            frame, crop_offset = self._apply_agent_view(frame, y, x, cell_size)

        # Draw trail
        if self.show_trail and len(self.trail) > 1:
            frame = self._draw_trail_alpha(frame, cell_size, step, crop_offset)

        return frame

    def finalize(self):
        """Save animation (GIF or MP4) using imageio."""
        if not self.save_video or not self.frames:
            return

        if self.as_gif:
            imageio.mimsave(self.video_path, self.frames, duration=50, loop=0)
            print(f"✓ Saved GIF to {self.video_path}")
        else:
            imageio.mimsave(self.video_path, self.frames, fps=20)
            print(f"✓ Saved MP4 to {self.video_path}")

        self.frames.clear()

    # ------------------------------------------------------------------
    #  Private helper methods (unchanged)
    # ------------------------------------------------------------------
    def _apply_fog_of_war(self, frame, grid_size, cell_size):
        text_height = self._text_height
        for yy in range(grid_size):
            for xx in range(grid_size):
                if (yy, xx) in self.visited:
                    continue
                y0 = yy * cell_size
                if y0 < text_height:
                    continue
                y1 = (yy + 1) * cell_size
                x0 = xx * cell_size
                x1 = (xx + 1) * cell_size
                frame[y0:y1, x0:x1] = (0, 0, 0)
        return frame

    def _apply_agent_view(self, frame, y, x, cell_size, view_size=3):
        win_size = cell_size * view_size
        half = win_size // 2
        cx = int((x + 0.5) * cell_size)
        cy = int((y + 0.5) * cell_size)
        top = cy - half
        bottom = cy + half
        left = cx - half
        right = cx + half

        cropped = np.zeros((win_size, win_size, 3), dtype=np.uint8)
        src_top = max(0, top)
        src_bottom = min(frame.shape[0], bottom)
        src_left = max(0, left)
        src_right = min(frame.shape[1], right)
        dst_top = src_top - top
        dst_bottom = dst_top + (src_bottom - src_top)
        dst_left = src_left - left
        dst_right = dst_left + (src_right - src_left)
        if dst_top < win_size and dst_left < win_size:
            cropped[dst_top:dst_bottom, dst_left:dst_right] = frame[src_top:src_bottom, src_left:src_right]
        return cropped, (left, top)

    def _draw_trail_alpha(self, frame, cell_size, current_step, crop_offset, trail_length=30):
        if len(self.trail) < 2:
            return frame

        left, top = crop_offset
        alpha_map = np.zeros(frame.shape[:2], dtype=np.uint8)

        for i in range(len(self.trail) - 1):
            y1, x1, s1 = self.trail[i]
            y2, x2, s2 = self.trail[i + 1]

            steps_ago = current_step - s1
            if steps_ago > trail_length:
                continue

            alpha = int(255 * np.exp(-steps_ago / (trail_length / 3)))
            if alpha == 0:
                continue

            p1 = (int((x1 + 0.5) * cell_size) - left, int((y1 + 0.5) * cell_size) - top)
            p2 = (int((x2 + 0.5) * cell_size) - left, int((y2 + 0.5) * cell_size) - top)
            cv2.line(alpha_map, p1, p2, alpha, thickness=max(1, cell_size // 20))

        alpha_norm = alpha_map.astype(np.float32) / 255.0
        alpha_norm = np.expand_dims(alpha_norm, axis=2)
        blended = (frame.astype(np.float32) * (1 - alpha_norm) + 255 * alpha_norm).astype(np.uint8)
        return blended