# src/networks/transformer.py
"""
Transformer-based policy network with positional encoding, causal masking,
recurrent memory tokens, and optional value head.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseNetwork, EmbeddingLayer, AttentionAggregator
from src.core.constants import VOCAB_SIZE, OBSERVATION_SIZE, ACTION_SIZE


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    Adds a unique vector to each position in the sequence,
    allowing the transformer to know the order of tokens.
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, D], returns x + positional encoding."""
        return x + self.pe[:, :x.size(1)]


class TransformerPolicyNet(BaseNetwork):
    """
    Transformer policy network with:
      - Causal self-attention (no looking into the future)
      - Learnable/sinusoidal positional encoding
      - Recurrent memory tokens that persist across calls
      - Optional auxiliary heads and value head
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 observation_size: int = OBSERVATION_SIZE,
                 hidden_size: int = 512,
                 action_size: int = ACTION_SIZE,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 memory_size: int = 10,
                 use_auxiliary: bool = False,
                 use_value_head: bool = False,
                 max_seq_len: int = 200):
        """
        Args:
            use_value_head: whether to include a value head (for PPO)
            max_seq_len: maximum sequence length for positional encoding
        """
        super().__init__(observation_size, action_size, hidden_size,
                         use_auxiliary, use_value_head)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.max_seq_len = max_seq_len

        # 1. Token embedding (within an observation)
        """
        Neural networks work with continuous vectors, not integers.
        Embedding maps each token (0-18) to a learned high-dimensional representation,
        allowing the network to capture similarities between tokens.
        The EmbeddingLayer also adds positional encoding inside the 10-token observation
        so that the model knows which token is the last action and which is energy.
        """
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)

        # 2. Aggregator: compress 10 token vectors into one per time step
        """
        The transformer expects a single vector per time step, not 10.
        This aggregator uses multi-head attention to combine the 10 token embeddings
        into one vector of size embed_dim. Attention is more flexible than the MLP
        aggregator in the LSTM because it can dynamically weigh important tokens.
        """
        self.aggregator = AttentionAggregator(embed_dim)

        # 3. Temporal positional encoding (across time steps)
        """
        The transformer is permutation-invariant. It treats the input sequence
        as a set unless we add positional information. This sinusoidal positional
        encoding gives each time step a unique signature, allowing the model
        to understand the order of observations.
        """
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len + memory_size)

        # 4. Transformer encoder with causal masking
        """
        A stack of self-attention and feedforward layers. The causal mask prevents
        each position from attending to future positions, which is critical for a
        policy network that must act without seeing the future.
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 5. Learnable memory tokens (initial state)
        """
        These are learnable vectors that act as a persistent memory across time steps.
        They are prepended to the sequence and the transformer can read and write to them.
        After each forward pass, the updated memory tokens are stored and used in the next call,
        turning the transformer into a recurrent model.
        """
        self.memory_tokens = nn.Parameter(
            torch.randn(1, memory_size, embed_dim) * 0.02
        )

        # 6. Policy head: maps processed observations to action logits
        """
        The hidden state (embed_dim) is already rich. A small MLP with layer norm,
        GELU, and dropout produces logits for the 6 actions.
        We apply the policy head only to the observation outputs, not to the memory tokens.
        """
        self.policy_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_size)
        )

        # 7. Auxiliary heads (optional)
        """
        They share the same transformer outputs but have their own small MLPs.
        The auxiliary losses shape the hidden state to be predictive of environment dynamics,
        independent of the reward signal.
        """
        if use_auxiliary:
            # Predicts the next scalar energy (continuous value)
            self.energy_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1)
            )
            # Predicts the next observation (10 tokens, treated as regression)
            self.observation_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, observation_size)
            )

        # Value head (for PPO) – created in BaseNetwork if use_value_head=True
        # (BaseNetwork already defines self.value_head)

        # Recurrent memory: stored between forward calls
        self._current_memory: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = 1):
        """
        Reset the recurrent memory to the initial learnable memory tokens.
        Called at the start of each episode.
        """
        device = next(self.parameters()).device
        self._current_memory = self.memory_tokens.expand(batch_size, -1, -1).detach()

    def forward(self,
                x: torch.Tensor,
                return_auxiliary: bool = False,
                return_value: bool = False) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Parameters
        ----------
        x : LongTensor [batch, seq_len, K]   (K = observation_size = 10)
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

        # Step 1: Embed and aggregate each observation's tokens
        """
        Each of the 10 tokens becomes a 512-dim vector; then attention aggregator
        compresses them into a single 512-dim vector per time step.
        """
        embedded = self.embedding(x)          # [B, T, K, D]
        aggregated = self.aggregator(embedded)  # [B, T, D]

        # Step 2: Retrieve or initialise recurrent memory with correct batch size
        """
        Ensure the stored memory matches the current batch size.
        If not, reinitialise it. This fixes issues when the batch size changes
        between forward calls (e.g., after resuming training with different batch size).
        """
        if self._current_memory is None or self._current_memory.size(0) != B:
            self.reset_state(B)
        memory = self._current_memory          # [B, memory_size, D]

        # Step 3: Concatenate memory tokens and observation sequence
        """
        We place the memory tokens at the beginning of the sequence.
        The causal mask ensures memory tokens can only attend to themselves
        (since nothing comes before them), while observations can attend to
        all previous observations and all memory tokens.
        """
        seq = torch.cat([memory, aggregated], dim=1)   # [B, memory_size + T, D]

        # Step 4: Add temporal positional encoding
        """
        Without this, the transformer would not know the order of the sequence.
        """
        seq = self.pos_encoder(seq)

        # Step 5: Apply causal mask (no looking into the future)
        """
        Create a square upper-triangular mask where True means "mask out".
        This ensures that for a position `i`, only positions `j <= i` are visible.
        """
        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        output = self.transformer(seq, mask=causal_mask)   # [B, seq_len, D]

        # Step 6: Split output back into memory and observation parts
        new_memory = output[:, :self.memory_size]          # [B, memory_size, D]
        obs_out = output[:, self.memory_size:]             # [B, T, D]

        # Step 7: Store updated memory for the next call (recurrent behaviour)
        self._current_memory = new_memory.detach()

        # Step 8: Policy logits
        logits = self.policy_head(obs_out)                 # [B, T, A]

        # Build output tuple (order: logits, [energy, obs], [value])
        outputs = [logits]

        if return_auxiliary and self.use_auxiliary:
            energy_pred = self.energy_head(obs_out)
            obs_pred = self.observation_head(obs_out)
            outputs.extend([energy_pred, obs_pred])

        if return_value and self.use_value_head:
            value = self.value_head(obs_out)
            outputs.append(value)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': VOCAB_SIZE,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'memory_size': self.memory_size,
            'max_seq_len': self.max_seq_len,
            'dropout': self.transformer.layers[0].dropout.p,
        })
        return config