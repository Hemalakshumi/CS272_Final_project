## custom_cnn.py
##%%writefile custom_cnn.py #if running in colab 
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # should be 4 stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2),  # -> (32, 62, 30)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),                # -> (64, 30, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),               # -> (128, 14, 6)
            nn.ReLU(),
            nn.Flatten()
        )

        # compute CNN output size dynamically
        with th.no_grad():
            dummy = th.zeros(1, *observation_space.shape)
            n_flat = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))
