import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PreventionEnv(gym.Env):
    """
    Custom Gym environment for RL-based network traffic prevention.
    Action Space:
        0 = Allow (let traffic through)
        1 = Block (prevent traffic)
    """

    def __init__(self, X, y):
        super(PreventionEnv, self).__init__()

        self.X = X.values if hasattr(X, "values") else X
        self.y = y.values if hasattr(y, "values") else y
        self.n_samples = len(self.X)
        self.current_index = 0

        # === Environment limits ===
        self.max_steps = 100  # Max steps per episode
        self.steps_taken = 0

        # === Define Spaces ===
        self.action_space = spaces.Discrete(2)  # 0 = Allow, 1 = Block
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.X.shape[1],),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = np.random.randint(0, self.n_samples)
        self.steps_taken = 0
        return self.X[self.current_index].astype(np.float32), {}

    def step(self, action):
        label = self.y[self.current_index]  # Ground truth

        # === Reward Design ===
        if action == 1 and label == 1:     # Blocked attack
            reward = +1.5
        elif action == 1 and label == 0:   # Blocked benign (False Positive)
            reward = -1.5
        elif action == 0 and label == 0:   # Allowed benign
            reward = +1.0
        else:                              # Allowed attack (False Negative)
            reward = -2.0

        # === Next Step ===
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        self.current_index = (self.current_index + 1) % self.n_samples
        next_state = self.X[self.current_index].astype(np.float32)

        return next_state, reward, done, False, {}
