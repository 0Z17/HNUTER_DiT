"""Experimental velocity-conditioned SDF DiT; transferred frozen map encoder."""
import numpy as np
import torch
from torch import nn
from .sdf_model import SDFPoseDiT
from .model import DiffusionSchedule

class MovingSDFDiT(SDFPoseDiT):
    def __init__(self):
        super().__init__()
        self.condition=nn.Sequential(nn.Linear(17,128),nn.SiLU(),nn.Linear(128,128))
        self.order=nn.Parameter(torch.randn(1,8,128)*.02)
        self.mean=torch.zeros(8,6);self.std=torch.ones(8,6)
        self.register_buffer('knot_times',torch.arange(8)*.5+.6)
    def initialize_from(self,base):
        source=base.state_dict();own=self.state_dict()
        self.load_state_dict({k:v for k,v in source.items() if k in own and own[k].shape==v.shape},strict=False)
        with torch.no_grad():
            self.condition[0].weight.zero_();self.condition[0].weight[:,:3].copy_(base.condition[0].weight[:,:3])
            self.condition[0].bias.copy_(base.condition[0].bias)
        self.freeze_encoder()
    def freeze_encoder(self):
        for name in ('cnn','spatial','age','history'):
            for p in getattr(self,name).parameters():p.requires_grad_(False)
    def baseline(self,c):
        position=c[:,None,3:6]*self.knot_times[None,:,None]
        return torch.cat((position,torch.zeros_like(position)),dim=-1)

@torch.inference_mode()
def moving_sample(model,condition,memory,seed=23,steps=20):
    schedule=DiffusionSchedule.cosine(100,condition.device)
    generator=torch.Generator(device=condition.device).manual_seed(seed)
    x=torch.randn((len(condition),8,6),device=condition.device,generator=generator)
    indices=torch.linspace(99,0,steps,device=condition.device).round().long()
    for i,t in enumerate(indices):
        v=model.forward_cached(x,t.expand(len(condition)),condition,memory);a=schedule.alpha_bars[t]
        clean=a.sqrt()*x-(1-a).sqrt()*v;noise=(1-a).sqrt()*x+a.sqrt()*v
        x=clean if i+1==len(indices) else schedule.alpha_bars[indices[i+1]].sqrt()*clean+(1-schedule.alpha_bars[indices[i+1]]).sqrt()*noise
    return model.denormalized(x,condition)
