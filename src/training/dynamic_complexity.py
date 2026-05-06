"""
Dynamic complexity manager – exact original implementation from monolithic trainer.py
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional

class ComplexityManager:
    """Manages dynamic complexity adjustment based on agent performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        
        # Dynamic complexity settings
        self.enabled = self.training_config['dynamic_complexity']
        self.performance_window = self.training_config['performance_window']
        self.increase_threshold = self.training_config['complexity_increase_threshold']
        self.decrease_threshold = self.training_config['complexity_decrease_threshold']
        self.complexity_step = self.training_config['complexity_step']
        self.min_complexity = self.training_config['min_complexity']
        self.max_complexity = self.training_config['max_complexity']
        self.adjustment_interval = self.training_config['adjustment_interval']
        self.stagnation_switch_interval = self.training_config['stagnation_switch_interval']
        self.stagnation_termination = self.training_config['stagnation_termination']
        self.min_basic_complexity = self.training_config['min_basic_complexity']
        self.curriculum_stages = self.training_config['curriculum_stages']
        
        # Current state - each stage maintains its own complexity
        self.current_stage_idx = 0
        self.stage_complexities = {stage: 0.0 for stage in self.curriculum_stages}
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.performance_window)
        self._last_performance_score = 0.0
        self.max_rewards_by_stage = {}
        self.epochs_without_progress = 0
        self.last_complexity_increase_epoch = 0
        self.last_max_reward = -float('inf')
        
        # Stage switching tracking
        self.stage_selection_counts = {stage: 0 for stage in self.curriculum_stages}
        self.total_switches = 0
        self.linear_cycle_complete = False
        
        # Statistics
        self.adjustments_made = 0
        self.stage_switches = 0

    def add_performance(self, reward: float):
        """Add performance metric to history"""
        self.performance_history.append(reward)
        
        # Check for progress in current stage
        current_max = self.max_rewards_by_stage.get(self.current_stage_idx, reward)
        if reward > current_max:
            self.max_rewards_by_stage[self.current_stage_idx] = reward
            self.epochs_without_progress = 0
            self.last_max_reward = reward
        else:
            self.epochs_without_progress += 1
    
    def get_current_task_class(self) -> str:
        """Get current task class based on curriculum stage"""
        if not self.enabled or self.current_stage_idx >= len(self.curriculum_stages):
            return self.config['environment']['task_class']
        return self.curriculum_stages[self.current_stage_idx]
    
    def get_current_complexity(self) -> float:
        """Get current complexity level"""
        if not self.enabled:
            return self.config['environment']['complexity_level']
        current_stage = self.get_current_task_class()
        return self.stage_complexities[current_stage]
    
    def should_adjust(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        if epoch % self.adjustment_interval != 0:
            return False
        if len(self.performance_history) < self.performance_window // 2:
            return False
        return True

    def should_switch_stage(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        current_stage = self.get_current_task_class()
        current_complexity = self.stage_complexities[current_stage]
        if current_stage == 'basic' and current_complexity < self.min_basic_complexity:
            return False
        if epoch - self.last_complexity_increase_epoch >= self.stagnation_switch_interval:
            return True
        if len(self.performance_history) >= self.performance_window:
            recent_std = np.std(list(self.performance_history)[-self.performance_window:])
            recent_mean = np.mean(list(self.performance_history)[-self.performance_window:])
            if recent_std < 0.1 and recent_mean < self.decrease_threshold:
                return True
        return False

    def calculate_performance_score(self) -> float:
        if not self.performance_history:
            return self._last_performance_score
        avg_performance = np.mean(list(self.performance_history)[-self.performance_window:])
        stage_idx = self.current_stage_idx
        if stage_idx not in self.max_rewards_by_stage:
            self.max_rewards_by_stage[stage_idx] = avg_performance
        else:
            self.max_rewards_by_stage[stage_idx] = max(self.max_rewards_by_stage[stage_idx], avg_performance)
        max_reward = self.max_rewards_by_stage[stage_idx]
        if max_reward < 0.1:
            return 0.0
        score = max(0.0, min(1.0, avg_performance / max_reward))
        self._last_performance_score = score
        return score

    def switch_to_next_stage(self, epoch: int) -> Dict[str, Any]:
        old_stage_idx = self.current_stage_idx
        old_stage = self.curriculum_stages[old_stage_idx]
        old_complexity = self.stage_complexities[old_stage]

        if not self.linear_cycle_complete:
            next_idx = (old_stage_idx + 1) % len(self.curriculum_stages)
            self.current_stage_idx = next_idx
            new_stage = self.curriculum_stages[next_idx]
            new_complexity = self.stage_complexities[new_stage]
            if next_idx == 0:
                self.linear_cycle_complete = True
            reason = "Linear progression (first cycle)"
            probs = None
        else:
            epsilon = 0.1
            weights = []
            for stage in self.curriculum_stages:
                complexity = self.stage_complexities[stage]
                freq = self.stage_selection_counts[stage] / (self.total_switches + 1)
                weight = (1.0 - complexity + epsilon) * (1.0 - freq + epsilon)
                weights.append(weight)
            probs = np.array(weights) / np.sum(weights)
            next_idx = np.random.choice(len(self.curriculum_stages), p=probs)
            self.current_stage_idx = next_idx
            new_stage = self.curriculum_stages[next_idx]
            new_complexity = self.stage_complexities[new_stage]
            reason = f"Probabilistic (complexity={new_complexity:.2f}, freq={self.stage_selection_counts[new_stage]/(self.total_switches+1):.2f})"

        self.stage_selection_counts[new_stage] += 1
        self.total_switches += 1
        self.epochs_without_progress = 0
        self.last_complexity_increase_epoch = epoch
        self.last_max_reward = -float('inf')
        self.performance_history.clear()
        self.stage_switches += 1

        adjustment_info = {
            "action": "switched_stage",
            "old_stage": old_stage,
            "new_stage": new_stage,
            "old_complexity": old_complexity,
            "new_complexity": new_complexity,
            "reason": reason,
            "stage_probs": probs.tolist() if probs is not None else None
        }
        return adjustment_info

    def adjust_complexity(self, epoch: int) -> Optional[Dict[str, Any]]:
        if not self.should_adjust(epoch):
            return None
        
        if self.should_switch_stage(epoch):
            adjustment_info = self.switch_to_next_stage(epoch)
            self.adjustments_made += 1
            return adjustment_info
        
        performance_score = self.calculate_performance_score()
        current_stage = self.get_current_task_class()
        old_complexity = self.stage_complexities[current_stage]

        if self.decrease_threshold <= performance_score <= self.increase_threshold:
            return None
        
        adjustment_info = {
            "performance_score": performance_score,
            "old_complexity": old_complexity,
            "old_stage": current_stage,
            "new_stage": current_stage,
        }
        
        if performance_score > self.increase_threshold:
            if old_complexity >= self.max_complexity:
                return None
            new_complexity = min(self.max_complexity, old_complexity + self.complexity_step)
            adjustment_info["action"] = "increased_complexity"
            self.epochs_without_progress = 0
            self.last_complexity_increase_epoch = epoch
        elif performance_score < self.decrease_threshold:
            if old_complexity <= self.min_complexity:
                return None
            new_complexity = max(self.min_complexity, old_complexity - self.complexity_step)
            adjustment_info["action"] = "decreased_complexity"
        else:
            return None

        adjustment_info["new_complexity"] = new_complexity
        self.stage_complexities[current_stage] = new_complexity
        self.adjustments_made += 1
        self.performance_history.clear()
        return adjustment_info

    def get_environment_config(self) -> Dict[str, Any]:
        env_config = self.config['environment'].copy()
        if self.enabled:
            current_stage = self.get_current_task_class()
            env_config['task_class'] = current_stage
            env_config['complexity_level'] = self.stage_complexities[current_stage]
            if current_stage == 'basic':
                env_config['n_doors'] = 0
                env_config['n_buttons_per_door'] = 0
                env_config['button_break_probability'] = 0.0
            elif current_stage == 'doors':
                env_config['n_doors'] = None
                env_config['n_buttons_per_door'] = 0
                env_config['button_break_probability'] = 0.0
            elif current_stage == 'buttons':
                env_config['n_doors'] = None
                env_config['n_buttons_per_door'] = None
                env_config['button_break_probability'] = None
            elif current_stage == 'complex':
                env_config['n_doors'] = None
                env_config['n_buttons_per_door'] = None
                env_config['button_break_probability'] = None
        return env_config
    
    def get_status(self) -> Dict[str, Any]:
        current_stage = self.get_current_task_class()
        return {
            "enabled": self.enabled,
            "current_stage": current_stage,
            "current_complexity": self.stage_complexities[current_stage],
            "all_stage_complexities": self.stage_complexities,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.curriculum_stages),
            "performance_history_size": len(self.performance_history),
            "adjustments_made": self.adjustments_made,
            "stage_switches": self.stage_switches,
            "epochs_without_progress": self.epochs_without_progress,
            "performance_score": self.calculate_performance_score(),
            "max_rewards_by_stage": self.max_rewards_by_stage,
            "basic_min_complexity_reached": self.stage_complexities['basic'] >= self.min_basic_complexity,
            "linear_cycle_complete": self.linear_cycle_complete,
            "stage_selection_counts": self.stage_selection_counts,
            "total_switches": self.total_switches
        }