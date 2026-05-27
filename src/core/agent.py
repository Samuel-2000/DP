# src/core/agent.py
"""
Trained agent using neural network policy.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from src.networks.lstm import LSTMPolicyNet
from src.networks.transformer import TransformerPolicyNet
from src.core.utils import safe_load
from src.core.constants import (
    OBSERVATION_SIZE, ACTION_SIZE, VOCAB_SIZE,
    ObservationTokens
)
from src.core.agent_base import BaseAgent


class Agent(BaseAgent):
    def __init__(self,
            network_type: str = 'lstm',
            observation_size: int = OBSERVATION_SIZE,
            action_size: int = ACTION_SIZE,
            hidden_size: int = 512,
            use_auxiliary: bool = False,
            use_value_head: bool = False,
            device: str = 'auto'
        ):
        super().__init__()
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.network_type = network_type
        self.use_auxiliary = use_auxiliary
        self.use_value_head = use_value_head
        
        self._validate_observation_range()
        
        if network_type == 'lstm':
            self.network = LSTMPolicyNet(
                vocab_size=VOCAB_SIZE,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                use_auxiliary=use_auxiliary,
                use_value_head=use_value_head
            )
        elif network_type == 'transformer':
            self.network = TransformerPolicyNet(
                vocab_size=VOCAB_SIZE,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                num_heads=8,
                num_layers=3,
                memory_size=10,
                use_auxiliary=use_auxiliary,
                use_value_head=use_value_head
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        self.network.to(self.device)
        print(f"Created {network_type} agent:")
        print(f"  Observation size: {observation_size}")
        print(f"  Action size: {action_size}")
        print(f"  Vocab size: {VOCAB_SIZE} (tokens 0-{VOCAB_SIZE-1})")
        print(f"  Device: {device}")
    
    def _validate_observation_range(self):
        max_token = ObservationTokens.ENERGY_LEVEL_4
        if max_token != VOCAB_SIZE - 1:
            raise ValueError(f"Observation token range mismatch: "
                           f"max_token={max_token}, VOCAB_SIZE={VOCAB_SIZE}")
        print(f"✓ Observation tokens valid: 0-{max_token}")
    
    def get_action(self, observation: np.ndarray) -> int:
        """Get action from neural network (deterministic for testing)."""
        with torch.no_grad():
            obs_min, obs_max = observation.min(), observation.max()
            if obs_min < 0 or obs_max >= VOCAB_SIZE:
                raise ValueError(f"Observation out of range [0, {VOCAB_SIZE-1}]: "
                               f"min={obs_min}, max={obs_max}")
            
            obs_tensor = torch.from_numpy(observation).long()
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.network(obs_tensor).squeeze(1)
            action = logits.argmax(dim=-1).item()
            
            if not (0 <= action < ACTION_SIZE):
                raise ValueError(f"Invalid action: {action}")
            return action
    
    def reset(self):
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()
    
    def save(self, path: str, extra_data: Dict[str, Any] = None):
        """Save agent to file, optionally with extra metadata."""
        config = {
            'network_type': self.network_type,
            'use_auxiliary': self.use_auxiliary,
            'hidden_size': self.network.hidden_size,
        }
        if hasattr(self.network, 'get_config'):
            config.update(self.network.get_config())
        else:
            config.update({
                'vocab_size': VOCAB_SIZE,
                'observation_size': OBSERVATION_SIZE,
                'action_size': ACTION_SIZE,
            })

        save_dict = {
            'state_dict': self.network.state_dict(),
            'config': config,
        }
        if extra_data is not None:
            save_dict.update(extra_data)

        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = safe_load(path, map_location=device)
        
        # First try loading as a checkpoint (saved by trainer)
        if 'model_state_dict' in checkpoint and 'model_config' in checkpoint:
            config = checkpoint['model_config']
            agent = cls(
                network_type=config['network_type'],
                observation_size=config['observation_size'],
                action_size=config['action_size'],
                hidden_size=config['hidden_size'],
                use_auxiliary=config['use_auxiliary'],
                use_value_head=config['use_value_head'],
                device=device
            )
            agent.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded agent from checkpoint {path} (strict=False)")
            return agent
        
        # Otherwise assume it's a standard agent file
        config = checkpoint['config']
        if 'model' in config:
            cfg = config['model']
        else:
            cfg = config
        
        agent = cls(
            network_type=cfg['network_type'],
            observation_size=cfg['observation_size'],
            action_size=cfg['action_size'],
            hidden_size=cfg['hidden_size'],
            use_auxiliary=config.get('use_auxiliary', False),
            use_value_head=config.get('use_value_head', False),
            device=device
        )
        agent.network.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded agent from {path} (strict=False)")
        return agent