from .nn_moe import NNMOE
from .weighted_moe import WeightedMOE

class MOEasModel:
    def __init__(self, moe_model, experts, context, obs_mixing=False):
        self.moe = moe_model
        self.experts = experts
        self.context = context
        self.obs_mixing = obs_mixing

    def predict(self, obs):
        context = (self.context, obs) if self.obs_mixing else (self.context,)
        exp_idx = self.moe(*context).argmax().item()
        return self.experts[exp_idx].predict(obs)