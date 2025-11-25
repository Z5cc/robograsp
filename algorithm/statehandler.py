import torch

from CONSTANTS import DEVICE, C


class StateHandler:
    def __init__(self):
        pass

    def initiate_state(self, obs): # [H,W]
        obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        state = obs[None,None,:,:].repeat(1,C,1,1)
        return state # [1,C,H,W]

    def update_state(self, state, obs): # [1,C,H,W] [H,W]
        obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        state = torch.roll(state, shifts=-1, dims=1)
        state[:,-1] = obs[None,:,:] # with state[:,-1] dimension of state gets reduced to [1,H,W]
        return state # [1,C,H,W]
