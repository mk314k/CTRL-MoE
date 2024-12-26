import torch


class NNMOE:
    def __init__(self, experts_context, device=torch.device('cpu')):
        super().__init__()
        self.experts_ctx = experts_context
        self.device = device

    def __call__(self, context):
        """_summary_

        Args:
            context (_type_): _description_
            obs (_type_): _description_

        Returns:
            _type_: _description_
        """
        ctx = context.as_tensor(device=self.device)
        sim = []
        for ectx in self.experts_ctx:
            tectx = ectx.as_tensor(device=self.device)
            sim.append((tectx * ctx).sum())

        return torch.tensor(sim).argmax().item()