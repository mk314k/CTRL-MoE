import torch
import torch.nn as nn

class ExpertPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(ExpertPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.fc(state)
