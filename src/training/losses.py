"""
Loss functions for RL training – REINFORCE, auxiliary, and PPO.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class PolicyLoss:
    """
    REINFORCE with baseline (normalized returns) and entropy bonus.
    This is the original method (your own).
    """

    def __init__(self, gamma: float = 0.97, entropy_coef: float = 0.01,
                 normalize_advantages: bool = True):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.normalize_advantages = normalize_advantages

    def _compute_returns(self, rewards: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute discounted returns (Monte Carlo) for each step."""
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return
            if mask is not None:
                running_return = running_return * mask[:, t]
        return returns

    def _compute_advantages(self, returns: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Advantages = returns, then normalized across batch and time."""
        advantages = returns.clone()
        if self.normalize_advantages:
            if mask is not None:
                masked_returns = returns * mask
                valid_count = mask.sum()
                if valid_count > 0:
                    mean = masked_returns.sum() / valid_count
                    std = torch.sqrt((masked_returns - mean).pow(2).sum() / valid_count + 1e-8)
                    advantages = (advantages - mean) / (std + 1e-8)
            else:
                mean = returns.mean()
                std = returns.std()
                advantages = (advantages - mean) / (std + 1e-8)
        return advantages

    def __call__(self, logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy loss and entropy value.

        Args:
            logits: [B, T, A] action logits
            actions: [B, T] action indices taken
            rewards: [B, T] immediate rewards
            mask: [B, T] where 1 indicates valid step (not terminal/padded)

        Returns:
            total_loss (scalar), mean_entropy (scalar)
        """
        B, T, A = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        returns = self._compute_returns(rewards, mask)
        advantages = self._compute_advantages(returns, mask)

        if mask is not None:
            action_log_probs = action_log_probs * mask
            advantages = advantages * mask
            valid_count = mask.sum()
        else:
            valid_count = B * T

        policy_loss = -(action_log_probs * advantages.detach()).sum() / valid_count

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)
        if mask is not None:
            entropy = entropy * mask
        entropy_loss = -self.entropy_coef * entropy.sum() / valid_count

        total_loss = policy_loss + entropy_loss
        return total_loss, entropy.sum() / valid_count


class AuxiliaryLoss:
    """
    Auxiliary losses for self-supervised learning:
      - Energy prediction (MSE)
      - Observation prediction (MSE on the 10-token vector)
    """

    def __init__(self, energy_coef: float = 0.1, obs_coef: float = 0.05):
        self.energy_coef = energy_coef
        self.obs_coef = obs_coef

    def __call__(self, energy_pred: torch.Tensor, energy_target: torch.Tensor,
                 obs_pred: torch.Tensor, obs_target: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            energy_pred: [B, T, 1] predicted energy (continuous)
            energy_target: [B, T] true energy
            obs_pred: [B, T, obs_dim] predicted observation tokens (treated as floats)
            obs_target: [B, T, obs_dim] true observation tokens (as ints, cast to float)
            mask: [B, T] mask for valid steps
        """
        energy_loss = F.mse_loss(energy_pred.squeeze(-1), energy_target)
        obs_loss = F.mse_loss(obs_pred, obs_target.float())
        if mask is not None:
            valid_ratio = mask.sum() / (mask.numel() + 1e-8)
            energy_loss = energy_loss * valid_ratio
            obs_loss = obs_loss * valid_ratio
        return self.energy_coef * energy_loss + self.obs_coef * obs_loss


class PPOLoss:
    def __init__(self, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, gamma=0.97, gae_lambda=0.95):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_gae(self, rewards, values, dones, mask):
        """
        rewards, values, dones, mask: all [B, T] (2D)
        Returns advantages, returns [B, T]
        """
        B, T = rewards.shape
        device = rewards.device
        gamma, gae_lambda = self.gamma, self.gae_lambda
        gamma_lambda = gamma * gae_lambda

        # TD error: δ_t = r_t + γ * V_{t+1} - V_t, with V_{T}=0
        next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, device=device)], dim=1)
        delta = rewards + gamma * next_values - values
        delta = delta * mask

        advantages = torch.zeros_like(delta)
        gae = torch.zeros(B, device=device)
        for t in range(T - 1, -1, -1):
            gae = delta[:, t] + gamma_lambda * (1 - dones[:, t]) * gae * mask[:, t]
            advantages[:, t] = gae

        returns = advantages + values
        return advantages, returns

    def __call__(self, logits, old_logits, actions, advantages, returns, values, mask):
        """
        Handles both 2D [N, A] and 3D [B, T, A] inputs.
        """
        # Flatten 3D to 2D for loss calculation
        if logits.dim() == 3:
            B, T, A = logits.shape
            logits = logits.view(-1, A)
            old_logits = old_logits.view(-1, A)
            actions = actions.view(-1)
            advantages = advantages.view(-1)
            returns = returns.view(-1)
            values = values.view(-1)
            mask = mask.view(-1)
        else:
            B = logits.size(0)

        # --- Policy loss (clipped surrogate) ---
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_action_log_probs = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(action_log_probs - old_action_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        # --- Value loss ---
        value_loss = F.mse_loss(values, returns, reduction='none')

        # --- Entropy bonus ---
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)

        # Apply mask and average
        mask_float = mask.float()
        policy_loss = (policy_loss * mask_float).sum() / mask_float.sum()
        value_loss = (value_loss * mask_float).sum() / mask_float.sum()
        entropy = (entropy * mask_float).sum() / mask_float.sum()

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
        return total_loss, metrics