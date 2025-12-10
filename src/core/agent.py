# src/core/agent.py - UPDATED VERSION
"""
Agent class (Updated to support multiple network types)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

# Import network types
from src.networks.lstm import LSTMPolicyNet
from src.networks.transformer import TransformerPolicyNet
from src.networks.multimemory import MultiMemoryPolicyNet
from src.core.utils import safe_load

class Agent:
    """Agent that interacts with environment using a policy network"""
    
    def __init__(self,
                 network_type: str = 'lstm',
                 observation_size: int = 10,
                 action_size: int = 6,
                 hidden_size: int = 512,
                 use_auxiliary: bool = False,
                 device: str = 'auto'):
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.network_type = network_type
        self.use_auxiliary = use_auxiliary
        
        # Create network based on type
        if network_type == 'lstm':
            self.network = LSTMPolicyNet(
                vocab_size=20,  # Matches original
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size
            )
        elif network_type == 'transformer':
            self.network = TransformerPolicyNet(
                vocab_size=20,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                num_heads=8,
                num_layers=3,
                memory_size=10,
                use_auxiliary=use_auxiliary
            )
        elif network_type == 'multimemory':
            self.network = MultiMemoryPolicyNet(
                vocab_size=20,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                transformer_heads=8,
                transformer_layers=3,
                cache_size=50,
                use_auxiliary=use_auxiliary
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        self.network.to(self.device)
        
    def act(self, 
            observation: np.ndarray,
            training: bool = False) -> int:
        """Select action based on observation"""
        with torch.set_grad_enabled(training):
            # Convert observation to LongTensor for embedding layer
            obs_tensor = torch.from_numpy(observation).long()
            
            # Add batch and sequence dimensions [1, 1, obs_dim]
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
            obs_tensor = obs_tensor.to(self.device)
            
            # Get action logits
            logits = self.network(obs_tensor)
            
            # Remove sequence dimension [1, action_size]
            logits = logits.squeeze(1)
            
            if training:
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                # Take greedy action
                action = logits.argmax(dim=-1).item()
            
            return action
    
    def reset(self):
        """Reset agent state"""
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()
    
    def save(self, path: str):
        """Save agent to file"""
        # Get configuration from network
        config = {
            'network_type': self.network_type,
            'use_auxiliary': self.use_auxiliary,
        }
        
        # Get network-specific config
        if hasattr(self.network, 'get_config'):
            config.update(self.network.get_config())
        else:
            # Fallback to default values
            config.update({
                'vocab_size': 20,
                'embed_dim': 512,
                'observation_size': 10,
                'hidden_size': 512,
                'action_size': 6
            })
        
        torch.save({
            'state_dict': self.network.state_dict(),
            'config': config
        }, path)
    

    @classmethod
    def load(cls, path: str, device: str = 'auto'):
        """Load agent from file"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

        checkpoint = safe_load(path, map_location=device)

        config = checkpoint.get('config', {})
        
        # Get network type, default to 'lstm' for backward compatibility
        network_type = config.get('network_type', 'lstm')
        
        # Get network parameters from config
        vocab_size = config.get('vocab_size', 20)
        embed_dim = config.get('embed_dim', 512)
        observation_size = config.get('observation_size', 10)
        hidden_size = config.get('hidden_size', 512)
        action_size = config.get('action_size', 6)
        use_auxiliary = config.get('use_auxiliary', False)
        
        # Create agent
        agent = cls(
            network_type=network_type,
            observation_size=observation_size,
            action_size=action_size,
            hidden_size=hidden_size,
            use_auxiliary=use_auxiliary,
            device=device
        )
        
        # Load weights
        agent.network.load_state_dict(checkpoint['state_dict'])
        
        return agent
    
    def test(self, 
             env,
             episodes: int = 10,
             visualize: bool = False,
             save_video: bool = False) -> Dict[str, Any]:
        """Test agent performance"""
        self.network.eval()
        
        rewards = []
        success_flags = []
        steps_list = []
        
        if save_video:
            import cv2
            video_writer = None
        
        for episode in range(episodes):
            obs, info = env.reset()
            self.reset()
            
            episode_reward = 0
            steps = 0
            terminated = truncated = False
            
            frames = []
            
            while not (terminated or truncated) and steps < env.max_steps:
                # Get action
                action = self.act(obs, training=False)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                # Record frame if needed
                if visualize or save_video:
                    frame = env.render()
                    if save_video:
                        frames.append(frame)
                    if visualize:
                        cv2.imshow('Test', frame)
                        cv2.waitKey(1)
            
            # Save video if requested
            if save_video and frames:
                if video_writer is None:
                    h, w, _ = frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_path = f'results/videos/test_episode_{episode}.mp4'
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, 20.0, (w, h)
                    )
                
                for frame in frames:
                    video_writer.write(frame)
            
            rewards.append(episode_reward)
            success_flags.append(steps == env.max_steps)
            steps_list.append(steps)
        
        if save_video and video_writer is not None:
            video_writer.release()
        
        if visualize:
            cv2.destroyAllWindows()
        
        return {
            'rewards': rewards,
            'success_flags': success_flags,
            'steps': steps_list,
            'avg_reward': np.mean(rewards),
            'success_rate': np.mean(success_flags) * 100,
            'avg_steps': np.mean(steps_list),
            'std_reward': np.std(rewards)
        }