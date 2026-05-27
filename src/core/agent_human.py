# src/core/agent_human.py
"""
Human-controlled agent for playable mode

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import cv2
import numpy as np
from typing import Dict, Any
import time
from pathlib import Path

from src.core.constants import Actions, DEFAULT_RENDER_SIZE
from src.visualization.visualizer import Visualizer
from src.core.agent_base import BaseAgent


class HumanAgent(BaseAgent):
    """Human-controlled agent using keyboard input (now with non‑blocking toggle handling)."""

    def __init__(self):
        super().__init__()
        self.action_map = {
            ord('a'): Actions.LEFT,    ord('A'): Actions.LEFT,
            ord('d'): Actions.RIGHT,   ord('D'): Actions.RIGHT,
            ord('w'): Actions.UP,      ord('W'): Actions.UP,
            ord('s'): Actions.DOWN,    ord('S'): Actions.DOWN,
            ord(' '): Actions.STAY,
            ord('b'): Actions.BUTTON,  ord('B'): Actions.BUTTON,
            13: Actions.BUTTON,        # Enter key
            81: Actions.LEFT,          # Left arrow
            82: Actions.UP,            # Up arrow
            83: Actions.RIGHT,         # Right arrow
            84: Actions.DOWN,          # Down arrow
        }
        
        self.key_descriptions = {
            Actions.LEFT: "A / Left Arrow",
            Actions.RIGHT: "D / Right Arrow",
            Actions.UP: "W / Up Arrow",
            Actions.DOWN: "S / Down Arrow",
            Actions.STAY: "Space",
            Actions.BUTTON: "B / Enter"
        }

    def test(self, env, args) -> Dict[str, Any]:
        """
        Human play mode with non‑blocking key handling for toggles and actions.
        """
        rewards = []
        success_flags = []
        steps_list = []

        print("\n" + "="*60)
        print("HUMAN PLAY MODE STARTED")
        print("="*60)
        print(f"Task Class: {env.task_class}")
        print(f"Complexity Level: {env.complexity_level:.2f}")
        print(f"Max Steps: {env.max_steps}")
        print(f"Test epochs: {args.epochs}")
        print(f"Episodes per epoch: {args.reinforce_intra_epochs}")
        print("="*60)

        print("\n" + "="*50)
        print("CONTROLS:")
        print(f"  Move Left:    {self.key_descriptions[Actions.LEFT]}")
        print(f"  Move Right:   {self.key_descriptions[Actions.RIGHT]}")
        print(f"  Move Up:      {self.key_descriptions[Actions.UP]}")
        print(f"  Move Down:    {self.key_descriptions[Actions.DOWN]}")
        print(f"  Stay:         {self.key_descriptions[Actions.STAY]}")
        print(f"  Press Button: {self.key_descriptions[Actions.BUTTON]}")
        print("  Quit:         Q or Esc")
        print("="*50)

        self._print_visualization_controls()

        total_episodes = 0
        early_stop = False
        paused = False
        pomdp_overview = False

        target_frame_time = 1.0 / 20.0

        for epoch in range(args.epochs):
            if early_stop:
                break
            obs, info = env.reset()
            print(f"\n--- EPOCH {epoch+1}/{args.epochs}: New grid (Type: {env.task_class}, Complexity: {env.complexity_level:.2f}) ---")

            for ep_in_epoch in range(args.reinforce_intra_epochs):
                if early_stop:
                    break
                if ep_in_epoch > 0:
                    obs, info = env.soft_reset()
                    print("  Soft reset: same grid, new chance!")

                vid_name = f"human_{env.task_class}_comp_{env.complexity_level:.2f}_ep_{epoch}_{ep_in_epoch}"
                vid_path = Path("results/videos") / f"{vid_name}.{'gif' if args.as_gif else 'mp4'}" if args.save_video else None
                viz = Visualizer(env, args.save_video, vid_path, args.agent_view,
                                args.fog_of_war, args.show_trail, args.as_gif,
                                DEFAULT_RENDER_SIZE)

                print(f"\nEpisode {total_episodes + 1} (epoch {epoch+1}, episode {ep_in_epoch+1}/{args.reinforce_intra_epochs})")
                episode_reward = 0
                steps = 0
                terminated = truncated = False
                last_frame_time = time.perf_counter()

                while not (terminated or truncated) and steps < env.max_steps:
                    # Render frame
                    frame = viz.render(steps, show_text=not pomdp_overview)
                    if frame is not None:
                        if pomdp_overview:
                            frame = self._apply_pomdp_overview(frame, env, obs, viz.agent_view)
                        cv2.imshow('Human Play Mode', frame)
                        if args.save_video:
                            viz.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Non‑blocking key check – handles toggles, pause, quit, actions
                    key = cv2.waitKey(1) & 0xFF

                    # Toggles & global controls
                    if key == ord('f'):
                        viz.toggle_fog_of_war()
                    elif key == ord('v'):
                        viz.toggle_agent_view()
                    elif key == ord('t'):
                        viz.toggle_show_trail()
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'Paused' if paused else 'Resumed'}. Press P to toggle.")
                    elif key == ord('o'):
                        pomdp_overview = not pomdp_overview
                        print(f"POMDP overview: {'ON' if pomdp_overview else 'OFF'}")
                    elif key == ord('q') or key == 27:   # Q or Esc
                        early_stop = True
                        break

                    if paused:
                        # Frame rate limiting while paused
                        now = time.perf_counter()
                        elapsed = now - last_frame_time
                        if elapsed < target_frame_time:
                            time.sleep(target_frame_time - elapsed)
                        last_frame_time = time.perf_counter()
                        continue

                    # Action detection (only when not paused)
                    action = None
                    if key in self.action_map:
                        action = self.action_map[key]
                        action_name = Actions(action).name
                        print(f"Action selected: {action_name}")
                    elif key != 255:   # ignore no key pressed
                        print(f"Invalid key: {chr(key) if 0 <= key < 128 else key}. Try again.")

                    # If an action was chosen, execute it and break inner wait loop
                    if action is not None:
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        steps += 1
                        print(f"Step {steps}: {Actions(action).name}, Reward={reward:.2f}, Energy={info.energy:.1f}")

                        if terminated or truncated:
                            break   # episode ended, will exit outer while

                    # Frame rate limiting
                    now = time.perf_counter()
                    elapsed = now - last_frame_time
                    if elapsed < target_frame_time:
                        time.sleep(target_frame_time - elapsed)
                    last_frame_time = time.perf_counter()

                viz.finalize()

                rewards.append(episode_reward)
                success_flags.append(steps == env.max_steps)
                steps_list.append(steps)

                energy = info['energy'] if isinstance(info, dict) and 'energy' in info else 0.0
                print(f"\nEpisode finished: Reward={episode_reward:.2f}, Steps={steps}/{env.max_steps}, Final Energy={energy:.1f}")
                
                cv2.waitKey(1000)
                total_episodes += 1

        cv2.destroyAllWindows()

        return {
            'rewards': rewards,
            'success_flags': success_flags,
            'steps': steps_list,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'success_rate': np.mean(success_flags) * 100 if success_flags else 0,
            'avg_steps': np.mean(steps_list) if steps_list else 0,
            'std_reward': np.std(rewards) if rewards else 0,
            'total_episodes': total_episodes
        }