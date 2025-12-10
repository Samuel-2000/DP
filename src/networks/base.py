"""
Base classes for neural networks
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import abc
from src.core.utils import safe_load


class BaseNetwork(nn.Module, abc.ABC):
    """Base class for all networks"""
    
    def __init__(self, 
                 observation_size: int = 10,
                 action_size: int = 6,
                 hidden_size: int = 512,
                 use_auxiliary: bool = False):
        super().__init__()
        
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.use_auxiliary = use_auxiliary
        
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
    
    @abc.abstractmethod
    def reset_state(self, batch_size: int = 1):
        """Reset internal state"""
        pass
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'auto'):
        """Load model"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = safe_load(path, map_location=device)
        config = checkpoint['config']
        
        # Create instance
        instance = cls(**config)
        instance.load_state_dict(checkpoint['state_dict'])
        instance.to(device)
        
        return instance
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration"""
        return {
            'observation_size': self.observation_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'use_auxiliary': self.use_auxiliary,
        }


class EmbeddingLayer(nn.Module):
    """Embedding layer with positional encoding"""
    
    def __init__(self, vocab_size: int = 20, embed_dim: int = 512):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        embedded = self.embedding(x)  # [B, T, K, D]
        # Add positional encoding to token positions only
        return embedded + self.pos_encoding.pe[:, :embedded.size(2), :]  # [1, K, D]


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        return x + self.pe[:, :x.size(1)]


class AttentionAggregator(nn.Module):
    """Attention-based aggregation of observation tokens"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x [B, T, K, D] -> [B, T, D]"""
        B, T, K, D = x.shape
        
        # Reshape for attention
        x_flat = x.view(B * T, K, D)
        
        # Compute attention
        Q = self.query(x_flat)
        K_emb = self.key(x_flat)
        V = self.value(x_flat)
        
        attn = torch.softmax(torch.bmm(Q, K_emb.transpose(1, 2)) * self.scale, dim=-1)
        aggregated = torch.bmm(attn, V).sum(dim=1)
        
        return aggregated.view(B, T, D)


class MLPAggregator(nn.Module):
    """MLP-based aggregation of observation tokens"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x [B, T, K*D] -> [B, T, D]"""
        return self.mlp(x)