import torch

def weights_init(m):
    if isinstance(m, (torch.nn.Linear)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
