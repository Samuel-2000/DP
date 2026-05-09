from .reinforce_trainer import ReinforceTrainer
from .ppo_trainer import PPOTrainer

def Trainer(config: dict):
    """Factory that returns the appropriate trainer based on algorithm."""
    algo = config['training']['algorithm']
    if algo == 'reinforce':
        return ReinforceTrainer(config)
    elif algo == 'ppo':
        return PPOTrainer(config)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")