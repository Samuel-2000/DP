# src/core/agent_base.py
"""
Base class for agents with common visualisation and testing logic.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
from pathlib import Path

from src.visualization.visualizer import Visualizer
from src.core.constants import DEFAULT_RENDER_SIZE, VOCAB_SIZE


class BaseAgent:
    """Base class for both trained and human agents."""

    def __init__(self):
        pass

    def get_action(self, observation: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self):
        pass

    @staticmethod
    def _print_visualization_controls():
        print("\n=== Visualisation Controls (press while window focused) ===")
        print("  F : Toggle Fog of War")
        print("  V : Toggle Agent View (3x3 crop)")
        print("  T : Toggle Trail")
        print("  P : Pause/Resume")
        print("  O : Toggle POMDP Overview (show agent's observation)")
        print("  Q : Quit test early")
        print("===========================================================")

    @staticmethod
    def _apply_pomdp_overview(frame, env, observation, agent_view_enabled=False):
        """
        Apply POMDP overview effect.
        If agent_view_enabled is True, only draw a green border (no dimming, no tokens).
        Otherwise, dim everything except the 3x3 area and show observation tokens.
        """
        grid_size = env.grid_size
        y, x = env.agent_y, env.agent_x
        h, w = frame.shape[:2]
        cell_size = w // grid_size

        if agent_view_enabled:
            # When agent view is on, the whole frame is already the 3x3 area.
            # Just draw a green border and return (no dimming, no text).
            result = frame.copy()
            cv2.rectangle(result, (0, 0), (w-1, h-1), (0, 255, 0), 2)
            return result

        # Full maze view: dim everything except the 3x3 region around agent
        dimmed = cv2.convertScaleAbs(frame, alpha=0.5, beta=0)
        half = cell_size * 3 // 2
        cx = int((x + 0.5) * cell_size)
        cy = int((y + 0.5) * cell_size)
        top = max(0, cy - half)
        bottom = min(h, cy + half)
        left = max(0, cx - half)
        right = min(w, cx + half)

        dimmed[top:bottom, left:right] = frame[top:bottom, left:right]
        cv2.rectangle(dimmed, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw observation tokens (only when agent view is OFF)
        token_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE", "Act", "Energy"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        text_x = max(10, w - 180)
        text_y = 20
        line_height = 20
        max_lines = len(observation)

        if text_y + max_lines * line_height + 10 > h:
            text_y = h - max_lines * line_height - 10

        cv2.rectangle(dimmed, (text_x - 5, text_y - 15),
                     (w - 5, text_y + max_lines * line_height + 5),
                     bg_color, -1)

        for i, (name, val) in enumerate(zip(token_names, observation)):
            text = f"{name}: {val}"
            cv2.putText(dimmed, text, (text_x, text_y + i * line_height),
                       font, font_scale, color, thickness)

        return dimmed

    def test(self, env, args, model_name: Optional[str] = None, seed: int = 42) -> Dict[str, Any]:
        """
        Generic test loop for trained agents (deterministic actions, no blocking).
        HumanAgent overrides this because it needs blocking input.
        """
        all_rewards = []
        all_success_flags = []
        all_steps = []

        if model_name is None:
            model_name = "agent"
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')

        render_size = DEFAULT_RENDER_SIZE if (args.visualize or args.save_video) else 0
        total_episodes = 0

        self._print_visualization_controls()

        early_stop = False
        paused = False
        pomdp_overview = False

        # Framerate limiting only when visualisation is enabled
        if args.visualize:
            target_frame_time = 1.0 / 20.0
            last_frame_time = time.perf_counter()

        for epoch in range(args.epochs):
            if early_stop:
                break
            obs, info = env.reset()
            if isinstance(obs, list):
                obs = np.array(obs, dtype=np.int32)
            self.reset()

            print(f"\n--- Epoch {epoch+1}/{args.epochs}: New grid (Type: {env.task_class}, Complexity: {env.complexity_level:.2f}) ---")

            for ep_in_epoch in range(args.reinforce_intra_epochs):
                if early_stop:
                    break
                if ep_in_epoch > 0:
                    obs, info = env.soft_reset()
                    if isinstance(obs, list):
                        obs = np.array(obs, dtype=np.int32)

                vid_name = f"{clean_model_name}_{env.task_class}_comp_{env.complexity_level:.2f}_ep_{epoch}_{ep_in_epoch}"
                vid_path = Path("results/videos") / f"{vid_name}.{'gif' if args.as_gif else 'mp4'}" if args.save_video else None
                viz = Visualizer(env, args.save_video, vid_path, args.agent_view,
                                args.fog_of_war, args.show_trail, args.as_gif,
                                render_size=render_size)

                episode_reward = 0
                steps = 0
                terminated = truncated = False

                while not (terminated or truncated) and steps < env.max_steps:
                    if paused:
                        frame = viz.render(steps, show_text=not pomdp_overview)
                        if args.visualize and frame is not None:
                            if pomdp_overview:
                                frame = self._apply_pomdp_overview(frame, env, obs, viz.agent_view)
                            cv2.imshow('Test', frame)
                        key = cv2.waitKey(50) & 0xFF
                        if key == ord('f'):
                            viz.toggle_fog_of_war()
                        elif key == ord('v'):
                            viz.toggle_agent_view()
                        elif key == ord('t'):
                            viz.toggle_show_trail()
                        elif key == ord('p'):
                            paused = False
                            print("Resumed.")
                        elif key == ord('o'):
                            pomdp_overview = not pomdp_overview
                            print(f"POMDP overview: {'ON' if pomdp_overview else 'OFF'}")
                        elif key == ord('q'):
                            early_stop = True
                            break
                        continue

                    # Normal step
                    if obs.min() < 0 or obs.max() >= VOCAB_SIZE:
                        obs = np.clip(obs, 0, VOCAB_SIZE - 1)

                    action = self.get_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if isinstance(obs, list):
                        obs = np.array(obs, dtype=np.int32)
                    episode_reward += reward
                    steps += 1

                    if args.visualize or args.save_video:
                        frame = viz.render(steps, show_text=not pomdp_overview)
                        if frame is not None:
                            if pomdp_overview:
                                frame = self._apply_pomdp_overview(frame, env, obs, viz.agent_view)
                            if args.visualize:
                                cv2.imshow('Test', frame)
                            if args.save_video:
                                viz.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('f'):
                                viz.toggle_fog_of_war()
                            elif key == ord('v'):
                                viz.toggle_agent_view()
                            elif key == ord('t'):
                                viz.toggle_show_trail()
                            elif key == ord('p'):
                                paused = True
                                print("Paused. Press P to resume.")
                            elif key == ord('o'):
                                pomdp_overview = not pomdp_overview
                                print(f"POMDP overview: {'ON' if pomdp_overview else 'OFF'}")
                            elif key == ord('q'):
                                early_stop = True
                                break

                    # Framerate limiting only when visualisation is enabled
                    if args.visualize:
                        now = time.perf_counter()
                        elapsed = now - last_frame_time
                        if elapsed < target_frame_time:
                            time.sleep(target_frame_time - elapsed)
                        last_frame_time = time.perf_counter()

                viz.finalize()

                all_rewards.append(episode_reward)
                all_success_flags.append(steps == env.max_steps)
                all_steps.append(steps)
                total_episodes += 1

        if args.visualize:
            cv2.destroyAllWindows()

        return {
            'rewards': all_rewards,
            'success_flags': all_success_flags,
            'steps': all_steps,
            'avg_reward': np.mean(all_rewards) if all_rewards else 0,
            'success_rate': np.mean(all_success_flags) * 100 if all_success_flags else 0,
            'avg_steps': np.mean(all_steps) if all_steps else 0,
            'std_reward': np.std(all_rewards) if all_rewards else 0,
            'total_episodes': total_episodes
        }