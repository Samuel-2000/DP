# src/__init__.py
"""
Maze RL package
"""

__version__ = "1.0.0"
__author__ = "Samuel Kuchta"

"""
# Export main classes
from src.core.environment import GridMazeWorld
from src.core.agent import Agent
from src.core.utils import setup_logging, seed_everything

# Export networks
from src.networks.lstm import LSTMPolicyNet
from src.networks.transformer import TransformerPolicyNet
from src.networks.multimemory import MultiMemoryPolicyNet

# Export training components
from src.training.trainer import Trainer
from src.training.losses import PolicyLoss, AuxiliaryLoss

# Export evaluation components
from src.evaluation.benchmark import Benchmark
from src.evaluation.visualization import Visualizer
"""