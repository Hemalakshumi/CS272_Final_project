# custom_rollout_buffer.py
from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np

class CustomRolloutBuffer(RolloutBuffer):
    """
    Extends RolloutBuffer to store 'infos' for each transition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize infos buffer
        self.infos = np.zeros((self.buffer_size,), dtype=object)

    def add(self, obs, action, reward, episode_start, value, log_prob, info=None):
        """
        Adds a transition to the buffer.
        - obs, action, reward, episode_start, value, log_prob: as in RolloutBuffer
        - info: optional dict with extra info
        """
        # Store info safely
        self.infos[self.pos] = info if info is not None else {}
        # Call original RolloutBuffer.add
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get_infos(self, batch_indices):
        """
        Returns the infos corresponding to batch indices.
        """
        return [self.infos[i] if self.infos[i] is not None else {} for i in batch_indices]
