from stable_baselines3.dqn.dqn import DQN
import torch
from torch import nn

class SimpleMOE:
    def __init__(self, experts):
        """
        Initialize with paths to pre-trained expert models.
        """
        self.experts = {prob: DQN.load(path) for prob, path in experts.items()}
        self.probabilities = list(self.experts.keys())

    def select_expert(self, context):
        """
        Select the expert based on the context (probability in this case).

        Args:
            context (_type_): _description_

        Returns:
            _type_: _description_
        """
        # For simplicity, use the nearest probability
        closest_prob = min(self.probabilities, key=lambda x: abs(x - context))
        return self.experts[closest_prob]
    
    def __call__(self, context, obs):
        return self.select_expert(context)(obs)
    

class WeightedMOE(nn.Module):
    def __init__(self, context_size, num_expert, experts):
        super().__init__()
        self.gating_network = nn.Linear(context_size, num_expert)
        self.experts = [DQN.load(path) for prob, path in experts.items()]

    def forward(self, context, obs):
        """_summary_
        Args:
            context (_type_): _description_
            obs (_type_): _description_

        Returns:
            _type_: _description_
        """
       #exp_idx = torch.argmax(self.gating_network(context)).item()
       # return self.experts[exp_idx](obs)
    
        exp_idx = torch.argmax(self.gating_network(obs)).item()
        return self.experts[exp_idx](obs)