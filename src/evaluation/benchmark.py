"""
Benchmarking utilities for model evaluation
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.core.environment import GridMazeWorld
from src.core.agent import Agent
from src.core.utils import setup_logging, save_results


class Benchmark:
    """Benchmark multiple models"""
    
    def __init__(self, 
                 models_dir: str = "models",
                 output_dir: str = "results/benchmarks"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging("benchmark")
    
    def run(self,
            episodes_per_model: int = 20,
            env_config: Optional[Dict] = None,
            verbose: bool = True) -> pd.DataFrame:
        """
        Run benchmark on all models in directory
        
        Returns:
            DataFrame with benchmark results
        """
        # Default environment config
        if env_config is None:
            env_config = {
                'grid_size': 11,
                'max_steps': 100,
                'obstacle_fraction': 0.25,
                'n_food_sources': 4
            }
        
        # Find all model files
        model_files = list(self.models_dir.glob("*.pt"))
        
        if len(model_files) == 0:
            self.logger.warning(f"No model files found in {self.models_dir}")
            return pd.DataFrame()
        
        if verbose:
            print(f"Found {len(model_files)} models to benchmark")
        
        results = []
        
        for model_file in tqdm(model_files, desc="Benchmarking", disable=not verbose):
            try:
                model_results = self._benchmark_single(
                    model_path=model_file,
                    episodes=episodes_per_model,
                    env_config=env_config
                )
                
                model_results['model_file'] = model_file.name
                results.append(model_results)
                
                if verbose:
                    print(f"✓ {model_file.name}: "
                          f"reward={model_results['avg_reward']:.2f} ± {model_results['std_reward']:.2f}")
            
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_file.name}: {e}")
                if verbose:
                    print(f"✗ {model_file.name}: Failed - {e}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # Sort by average reward (descending)
            df = df.sort_values('avg_reward', ascending=False)
            
            # Save results
            self._save_results(df)
            
            # Generate plots
            self._generate_plots(df)
        
        return df
    
    def _benchmark_single(self,
                         model_path: Path,
                         episodes: int,
                         env_config: Dict) -> Dict[str, Any]:
        """Benchmark a single model"""
        # Load agent
        agent = Agent.load(str(model_path))
        
        # Create environment
        env = GridMazeWorld(**env_config)
        
        # Run episodes
        rewards = []
        success_flags = []
        steps_list = []
        energies = []
        
        for episode in range(episodes):
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            steps = 0
            terminated = truncated = False
            
            while not (terminated or truncated):
                action = agent.act(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            success_flags.append(steps == env.max_steps)  # Survived full episode
            steps_list.append(steps)
            energies.append(info.get('energy', 0))
        
        # Compute statistics
        results = {
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'success_rate': float(np.mean(success_flags) * 100),
            'avg_steps': float(np.mean(steps_list)),
            'std_steps': float(np.std(steps_list)),
            'avg_final_energy': float(np.mean(energies)),
            'num_episodes': episodes
        }
        
        # Add network info if available
        if hasattr(agent, 'network_type'):
            results['network_type'] = agent.network_type
        
        return results
    
    def _save_results(self, df: pd.DataFrame):
        """Save benchmark results"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.output_dir / f"benchmark_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for easier reading
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        df.to_json(json_path, orient='records', indent=2)
        
        # Save summary markdown
        md_path = self.output_dir / f"benchmark_{timestamp}.md"
        self._create_markdown_report(df, md_path)
        
        self.logger.info(f"Results saved to {csv_path}")
    
    def _create_markdown_report(self, df: pd.DataFrame, path: Path):
        """Create markdown report"""
        with open(path, 'w') as f:
            f.write("# Benchmark Results\n\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models: {len(df)}\n")
            f.write(f"Episodes per model: {df['num_episodes'].iloc[0] if len(df) > 0 else 0}\n\n")
            
            f.write("## Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Summary\n\n")
            if 'network_type' in df.columns:
                f.write("### By Network Type\n\n")
                for network_type in df['network_type'].unique():
                    type_df = df[df['network_type'] == network_type]
                    if len(type_df) > 0:
                        f.write(f"**{network_type.upper()}**\n")
                        f.write(f"- Count: {len(type_df)}\n")
                        f.write(f"- Best reward: {type_df['avg_reward'].max():.2f}\n")
                        f.write(f"- Average reward: {type_df['avg_reward'].mean():.2f}\n")
                        f.write(f"- Best success rate: {type_df['success_rate'].max():.1f}%\n\n")
            
            f.write("### Top 3 Models\n\n")
            for i, row in df.head(3).iterrows():
                f.write(f"{i+1}. **{row['model_file']}**\n")
                f.write(f"   - Reward: {row['avg_reward']:.2f} ± {row['std_reward']:.2f}\n")
                f.write(f"   - Success rate: {row['success_rate']:.1f}%\n")
                f.write(f"   - Steps: {row['avg_steps']:.1f} ± {row['std_steps']:.1f}\n")
                if 'network_type' in row:
                    f.write(f"   - Network type: {row['network_type']}\n")
                f.write("\n")
    
    def _generate_plots(self, df: pd.DataFrame):
        """Generate visualization plots"""
        if len(df) == 0:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Benchmark Results', fontsize=16)
        
        # 1. Reward distribution
        ax = axes[0, 0]
        if len(df) > 1:
            sns.boxplot(data=df, y='avg_reward', ax=ax)
        else:
            ax.bar([0], df['avg_reward'].values)
        ax.set_title('Reward Distribution')
        ax.set_ylabel('Average Reward')
        
        # 2. Success rate vs reward
        ax = axes[0, 1]
        scatter = ax.scatter(df['avg_reward'], df['success_rate'], 
                           alpha=0.6, s=100)
        ax.set_xlabel('Average Reward')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Reward vs Success Rate')
        
        # Add model names as annotations for top performers
        top_n = min(5, len(df))
        for i in range(top_n):
            ax.annotate(df.iloc[i]['model_file'][:15],  # Truncate name
                       (df.iloc[i]['avg_reward'], df.iloc[i]['success_rate']),
                       fontsize=8, alpha=0.7)
        
        # 3. Network type comparison (if available)
        ax = axes[0, 2]
        if 'network_type' in df.columns and len(df['network_type'].unique()) > 1:
            type_data = df.groupby('network_type').agg({
                'avg_reward': ['mean', 'std', 'count'],
                'success_rate': 'mean'
            }).round(2)
            
            x = range(len(type_data))
            ax.bar(x, type_data['avg_reward']['mean'], 
                  yerr=type_data['avg_reward']['std'], capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(type_data.index, rotation=45)
            ax.set_title('Performance by Network Type')
            ax.set_ylabel('Average Reward')
            
            # Add count labels
            for i, (idx, row) in enumerate(type_data.iterrows()):
                count = int(row['avg_reward']['count'])
                ax.text(i, row['avg_reward']['mean'] / 2, 
                       f"n={count}", ha='center', va='center', color='white')
        else:
            ax.text(0.5, 0.5, 'No network type data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Network Type Comparison')
        
        # 4. Reward histogram
        ax = axes[1, 0]
        ax.hist(df['avg_reward'], bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Average Reward')
        ax.set_ylabel('Count')
        ax.set_title('Reward Distribution Histogram')
        
        # 5. Steps vs reward
        ax = axes[1, 1]
        ax.scatter(df['avg_steps'], df['avg_reward'], alpha=0.6)
        ax.set_xlabel('Average Steps')
        ax.set_ylabel('Average Reward')
        ax.set_title('Steps vs Reward')
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['avg_steps'], df['avg_reward'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['avg_steps'].min(), df['avg_steps'].max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.8)
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = [
            "Summary Statistics",
            "===================",
            f"Total models: {len(df)}",
            f"Mean reward: {df['avg_reward'].mean():.2f}",
            f"Std reward: {df['avg_reward'].std():.2f}",
            f"Best reward: {df['avg_reward'].max():.2f}",
            f"Worst reward: {df['avg_reward'].min():.2f}",
            f"Mean success: {df['success_rate'].mean():.1f}%",
            "",
            "Top 3 Models:"
        ]
        
        for i in range(min(3, len(df))):
            model_name = df.iloc[i]['model_file'][:20]  # Truncate
            reward = df.iloc[i]['avg_reward']
            success = df.iloc[i]['success_rate']
            summary_text.append(f"{i+1}. {model_name}: {reward:.2f} ({success:.1f}%)")
        
        ax.text(0.1, 0.95, "\n".join(summary_text), 
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_path}")
    
    def compare_architectures(self,
                             architectures: List[str],
                             episodes_per_arch: int = 10,
                             trials: int = 3) -> pd.DataFrame:
        """
        Compare different architectures
        
        Args:
            architectures: List of architecture names to compare
            episodes_per_arch: Episodes per architecture per trial
            trials: Number of trials per architecture
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for arch in architectures:
            for trial in range(trials):
                # Create fresh agent with this architecture
                agent = Agent(
                    network_type=arch,
                    observation_size=10,  # Default observation size
                    action_size=6,       # Default action size
                    hidden_size=128      # Smaller for quick comparison
                )
                
                # Create environment
                env = GridMazeWorld(grid_size=11, max_steps=50)
                
                # Run evaluation
                rewards = []
                for episode in range(episodes_per_arch):
                    obs, info = env.reset()
                    agent.reset()
                    
                    episode_reward = 0
                    terminated = truncated = False
                    
                    while not (terminated or truncated):
                        action = agent.act(obs, training=False)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                    
                    rewards.append(episode_reward)
                
                # Record results
                results.append({
                    'architecture': arch,
                    'trial': trial + 1,
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'min_reward': np.min(rewards),
                    'max_reward': np.max(rewards),
                    'episodes': episodes_per_arch
                })
        
        df = pd.DataFrame(results)
        
        # Save comparison results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"architecture_comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate comparison plot
        self._plot_architecture_comparison(df)
        
        return df
    
    def _plot_architecture_comparison(self, df: pd.DataFrame):
        """Plot architecture comparison results"""
        if len(df) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Group by architecture
        grouped = df.groupby('architecture')
        
        # Create box plot
        data_to_plot = [group['avg_reward'].values for name, group in grouped]
        labels = [name.upper() for name in grouped.groups.keys()]
        
        plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Add individual points
        for i, (name, group) in enumerate(grouped):
            x = np.random.normal(i + 1, 0.04, size=len(group))
            plt.scatter(x, group['avg_reward'], alpha=0.6, color='red', s=50)
        
        plt.title('Architecture Comparison', fontsize=16)
        plt.xlabel('Architecture', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for i, (name, group) in enumerate(grouped):
            mean = group['avg_reward'].mean()
            std = group['avg_reward'].std()
            stats_text.append(f"{name.upper()}: {mean:.2f} ± {std:.2f}")
        
        plt.figtext(0.02, 0.02, "\n".join(stats_text), 
                   fontsize=9, verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"architecture_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()