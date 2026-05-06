# src/networks/lstm.py - FIXED VERSION with value head
"""
LSTM-based policy network (Simplified to match original)
Now includes optional value head for PPO.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseNetwork

from src.core.constants import VOCAB_SIZE, OBSERVATION_SIZE, ACTION_SIZE


class LSTMPolicyNet(BaseNetwork):  # Inherit from BaseNetwork
    """LSTM-based policy network with optional auxiliary and value heads"""

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 observation_size: int = OBSERVATION_SIZE,
                 hidden_size: int = 512,
                 action_size: int = ACTION_SIZE,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 use_auxiliary: bool = False,
                 use_value_head: bool = False):
        """
        Args:
            use_value_head: whether to include a value head (for PPO)
        """
        # Call parent constructor (now supports use_value_head)
        super().__init__(observation_size, action_size, hidden_size,
                         use_auxiliary, use_value_head)

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Token embedding
        """
        Neural networks work with continuous vectors, not integers. 
        Embedding maps each token to a learned high-dimensional representation, 
        allowing the network to capture similarities 
        (e.g., tokens for different energy levels might be close in embedding space).
        """
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=None,
        )

        # Learnable positional encodings for K tokens inside an observation
        """
        Without positional encoding, the network would treat the 10 tokens as a set,
        therefore it wouldn't know that token 8 is the last action and token 9 is energy.
        The LSTM processes tokens sequentially, but at the moment of aggregation (next step),
        we flatten all tokens. positional encoding preserves which token is where.
        """
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)

        # ConcatMLP-style aggregator
        """
        The LSTM expects a single input vector per time step, not 10 tokens. 
        The aggregator compresses the 10 token vectors into one informative vector. 
        It is an MLP that can learn non-linear interactions among tokens 
        (e.g., “if neighbor is a button and last action was BUTTON, then something important”).
        Alternative: Could use attention (as in transformer network) but MLP is simpler and works.
        """
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim * observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        """
        The environment is partially observable (agent sees only 3×3 neighbours).
        To know where it is and what happened before, the agent needs memory.
        The LSTM maintains a hidden state that summarises the entire history.
        It can remember which doors it opened, which buttons it pressed, and the layout it has explored.
        """
        # LSTM memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Policy head (logits) - OUTPUTS 6 ACTIONS
        """
        The hidden state is already a rich representation.
        A simple linear projection is sufficient to map it to action preferences,
        and it adds minimal parameters, reducing overfitting.
        """
        self.head = nn.Linear(hidden_size, action_size)

        # Auxiliary heads (if needed)
        """
        They share the same LSTM hidden state but have their own small MLPs to handle the different prediction tasks. 
        The auxiliary losses influence the LSTM training, shaping the hidden state to be more predictive (independent of reward).
        """
        if use_auxiliary:
            # Predicts the next scalar energy (redundant but provides extra signal, as observation_head has only 5 energy levels instead of float value).
            self.energy_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            # Predicts the next observation (10 tokens, treated as regression with MSE). This forces the hidden state to encode environment dynamics.
            self.observation_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, observation_size)
            )

        # Value head (for PPO) – created in BaseNetwork if use_value_head=True
        # (BaseNetwork already defines self.value_head)

        # Hidden state - store as None initially
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.current_batch_size: Optional[int] = None

    def reset_state(self, batch_size: int = 1):
        """Reset LSTM hidden state (call at start of each episode)."""
        self.hidden_state = None
        self.current_batch_size = None

    def forward(self,
                x: torch.Tensor,
                return_auxiliary: bool = False,
                return_value: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : LongTensor [batch, seq, K] (K = observation_size = 10)
        return_auxiliary : if True and use_auxiliary, return auxiliary predictions
        return_value : if True and use_value_head, return value predictions

        Returns
        -------
        - If neither auxiliary nor value: logits [B, T, A]
        - If only auxiliary: (logits, energy_pred, obs_pred)
        - If only value: (logits, value)
        - If both: (logits, energy_pred, obs_pred, value)
        """
        B, T, K = x.shape

        # Ensure input is LongTensor
        if x.dtype != torch.long:
            x = x.long()

        # Validate token range
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < 0 or x_max >= self.vocab_size:
            raise ValueError(f"Input tokens out of range [0, {self.vocab_size-1}]: "
                             f"min={x_min}, max={x_max}")

        # Embed tokens: [B, T, K, D]
        x_embed = self.embedding(x)

        # Add positional encoding
        x_embed = x_embed + self.pos_embed  # broadcast (K, D) -> (B, T, K, D)

        # Flatten and aggregate: [B, T, K*D] -> [B, T, H]
        x_flat = x_embed.view(B, T, -1)
        aggregated = self.aggregator(x_flat)

        # LSTM over the temporal dimension
        if self.hidden_state is None or self.current_batch_size != B:
            # Initialize hidden state with correct batch size
            h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)
            self.current_batch_size = B

        out, self.hidden_state = self.lstm(aggregated, self.hidden_state)  # [B, T, H]

        # Policy logits
        logits = self.head(out)  # [B, T, A]

        # Build output tuple (order: logits, [energy, obs], [value])
        outputs = [logits]

        if return_auxiliary and self.use_auxiliary:
            energy_pred = self.energy_head(out)
            obs_pred = self.observation_head(out)
            outputs.extend([energy_pred, obs_pred])

        if return_value and self.use_value_head:
            value = self.value_head(out)
            outputs.append(value)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def get_config(self):
        """Get configuration for saving"""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        })
        return config