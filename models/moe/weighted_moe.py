import torch
from torch import nn
import torch.nn.functional as F

class WeightedMOE(nn.Module):
    def __init__(self, context_size,  num_expert, device=torch.device('cpu')):
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(context_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_expert)
        ).to(device)
        self.device = device

    def forward(self, context):
        """_summary_

        Args:
            context (_type_): _description_
            obs (_type_): _description_

        Returns:
            _type_: _description_
        """
        ctx = context.as_tensor(device=self.device)
        return F.softmax(self.gating_network(ctx), dim=-1)