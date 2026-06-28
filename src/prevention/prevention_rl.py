"""
RL-Based Prevention Training

Train and deploy PPO agent for prevention decisions.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable_baselines3 not installed. RL training not available.")

from .env_prevention import PreventionEnv


class PreventionRL:
    """RL-based prevention using PPO agent."""
    
    def __init__(self, env=None, learning_rate=0.001):
        """
        Initialize RL prevention agent.
        
        Args:
            env: Gym environment (PreventionEnv)
            learning_rate: PPO learning rate
        """
        if not HAS_SB3:
            raise ImportError("stable_baselines3 is required for RL training")
        
        self.env = env
        self.agent = None
        self.learning_rate = learning_rate
    
    def train(self, X_train, y_train, X_test=None, y_test=None, total_timesteps=10000, verbose=1):
        """
        Train PPO agent for prevention.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            total_timesteps: Total training timesteps
            verbose: Verbosity level
            
        Returns:
            Trained agent
        """
        print("🚀 Training RL prevention agent...")
        
        # Create environment
        if self.env is None:
            self.env = PreventionEnv(X_train, y_train)
        
        # Check environment
        try:
            check_env(self.env)
            print("✅ Environment validation passed")
        except Exception as e:
            print(f"⚠️  Environment validation warning: {e}")
        
        # Create PPO agent
        self.agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            n_steps=64,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=verbose
        )
        
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        print("✅ RL agent training complete")
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test)
        
        return self.agent
    
    def predict(self, X):
        """
        Make prevention predictions using trained agent.
        
        Args:
            X: Features to predict on
            
        Returns:
            Actions (0=Allow, 1=Block)
        """
        if self.agent is None:
            raise ValueError("Agent not trained. Call train() first.")
        
        actions = []
        for i in range(len(X)):
            observation = X[i].astype(np.float32) if hasattr(X, '__getitem__') else X.iloc[i].values.astype(np.float32)
            action, _ = self.agent.predict(observation, deterministic=True)
            actions.append(action)
        
        return np.array(actions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate agent on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("📊 Evaluating RL agent...")
        
        predictions = self.predict(X_test)
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        
        # Metrics
        correct = (predictions == y_test_array).sum()
        accuracy = correct / len(y_test_array)
        
        tp = ((predictions == 1) & (y_test_array == 1)).sum()
        fp = ((predictions == 1) & (y_test_array == 0)).sum()
        fn = ((predictions == 0) & (y_test_array == 1)).sum()
        tn = ((predictions == 0) & (y_test_array == 0)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  True Pos:  {tp}, True Neg:  {tn}")
        print(f"  False Pos: {fp}, False Neg: {fn}")
    
    def save(self, save_path):
        """
        Save trained agent.
        
        Args:
            save_path: Path to save agent
        """
        if self.agent is None:
            raise ValueError("No agent to save. Train first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.agent.save(save_path)
        print(f"✅ Agent saved to: {save_path}")
    
    def load(self, load_path):
        """
        Load trained agent.
        
        Args:
            load_path: Path to load agent
        """
        if HAS_SB3:
            self.agent = PPO.load(load_path)
            print(f"✅ Agent loaded from: {load_path}")
        else:
            raise ImportError("stable_baselines3 required to load agent")
