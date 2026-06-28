"""
Prevention Environment for RL Training

Custom Gym environment for RL-based network traffic prevention.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PreventionEnv(gym.Env):
    """
    Custom Gym environment for RL-based network traffic prevention.
    
    Action Space:
        0 = Allow (let traffic through)
        1 = Block (prevent traffic)
        
    Reward:
        +1 for correctly blocking attacks
        +0.1 for correctly allowing normal traffic
        -1 for incorrectly allowing attacks (false negative)
        -0.5 for incorrectly blocking normal traffic (false positive)
    """
    
    def __init__(self, X, y, max_steps=100):
        """
        Initialize prevention environment.
        
        Args:
            X: Feature data (array-like)
            y: Labels (0=normal, 1=attack)
            max_steps: Maximum steps per episode
        """
        super(PreventionEnv, self).__init__()
        
        self.X = X.values if hasattr(X, "values") else X
        self.y = y.values if hasattr(y, "values") else y
        self.n_samples = len(self.X)
        self.current_index = 0
        self.max_steps = max_steps
        self.steps_taken = 0
        
        # Define spaces
        self.action_space = spaces.Discrete(2)  # 0 = Allow, 1 = Block
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.X.shape[1],),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.current_index = 0
        self.steps_taken = 0
        observation = self.X[self.current_index].astype(np.float32)
        return observation, {}
    
    def step(self, action):
        """
        Execute one step in environment.
        
        Args:
            action: 0=Allow, 1=Block
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps_taken += 1
        
        # Get reward based on action and true label
        true_label = self.y[self.current_index]
        
        if action == 1 and true_label == 1:
            # Correctly blocked attack
            reward = 1.0
        elif action == 0 and true_label == 0:
            # Correctly allowed normal traffic
            reward = 0.1
        elif action == 0 and true_label == 1:
            # False negative - allowed attack
            reward = -1.0
        else:
            # False positive - blocked normal traffic
            reward = -0.5
        
        # Move to next sample
        self.current_index += 1
        terminated = self.current_index >= self.n_samples
        truncated = self.steps_taken >= self.max_steps
        
        if terminated or truncated:
            observation = np.zeros(self.X.shape[1], dtype=np.float32)
        else:
            observation = self.X[self.current_index].astype(np.float32)
        
        info = {'true_label': true_label, 'action': action}
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment state."""
        if self.current_index > 0:
            idx = self.current_index - 1
            print(f"Step {self.steps_taken}: Sample {idx}, Label: {self.y[idx]}")
