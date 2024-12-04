import torch
import torch.nn as nn



class GatingNetwork(nn.Module):
    def __init__(self, context_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(context_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, context):
        return self.fc(context)
