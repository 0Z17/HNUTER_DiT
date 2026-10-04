"""Local nvblox distance-field history + existing DiT blocks, six-DoF output."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .model import DiTBlock,timestep_embedding,DiffusionSchedule
from .sdf_pose import PROGRESS,SCHEMA
from .sdf_mapping import GRID,CHANNELS


class SDFPoseDiT(nn.Module):
    def __init__(self,dimension=128,depth=4,heads=4):
        super().__init__(); self.architecture=dict(dimension=dimension,depth=depth,heads=heads)
        self.cnn=nn.Sequential(nn.Conv3d(len(CHANNELS),16,4,stride=4),nn.SiLU(),
            nn.Conv3d(16,48,3,stride=2,padding=1),nn.SiLU(),nn.Conv3d(48,dimension,1),nn.SiLU())
        axis=torch.arange(4)*2-3.5
        xyz=torch.stack(torch.meshgrid(axis+2,axis,axis,indexing='ij'),dim=-1).reshape(64,3)/4
        self.register_buffer('xyz',xyz,persistent=False)
        self.spatial=nn.Linear(3,dimension); self.age=nn.Sequential(nn.Linear(1,dimension),nn.SiLU(),nn.Linear(dimension,dimension))
        self.history=nn.TransformerEncoderLayer(dimension,heads,dimension*2,dropout=0.,batch_first=True,norm_first=True)
        self.condition=nn.Sequential(nn.Linear(6,dimension),nn.SiLU(),nn.Linear(dimension,dimension))
        self.time=nn.Sequential(nn.Linear(64,dimension),nn.SiLU(),nn.Linear(dimension,dimension))
        self.input=nn.Linear(6,dimension);self.order=nn.Parameter(torch.randn(1,6,dimension)*.02)
        self.blocks=nn.ModuleList([DiTBlock(dimension,heads,dimension,True) for _ in range(depth)])
        self.norm=nn.LayerNorm(dimension);self.output=nn.Linear(dimension,6)
        self.register_buffer('mean',torch.zeros(6,6));self.register_buffer('std',torch.ones(6,6))
        self.register_buffer('progress',torch.tensor(PROGRESS,dtype=torch.float32))
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight);nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def encode(self,fields,ages):
        if fields.ndim!=6 or tuple(fields.shape[1:])!=(4,len(CHANNELS),GRID,GRID,GRID):
            raise ValueError('invalid GPU local SDF history shape')
        b=len(fields); value=self.cnn(fields.reshape(b*4,len(CHANNELS),GRID,GRID,GRID).float())
        value=value.flatten(2).transpose(1,2).reshape(b,4,64,-1)
        value=value+self.spatial(self.xyz)[None,None]+self.age(ages[...,None])[:,:,None]
        return self.history(value.reshape(b,256,-1))

    def forward_cached(self,x,t,condition,memory):
        context=self.time(timestep_embedding(t,64))+self.condition(condition)
        value=self.input(x)+self.order
        mask=torch.ones(memory.shape[:2],device=value.device,dtype=torch.bool)
        for block in self.blocks:value=block(value,context,memory,mask,None)
        return self.output(self.norm(value))

    def baseline(self,target):return self.progress[None,:,None]*target[:,None]
    def normalized(self,free,target):return (free-self.baseline(target)-self.mean)/self.std
    def denormalized(self,value,target):return value*self.std+self.mean+self.baseline(target)


@torch.inference_mode()
def sample(model,fields,ages,target,seed=23,steps=20,ablation=None):
    if ablation=='empty':fields=torch.zeros_like(fields)
    elif ablation=='latest':fields=fields[:,-1:].expand_as(fields);ages=torch.zeros_like(ages)
    memory=model.encode(fields,ages)
    schedule=DiffusionSchedule.cosine(100,fields.device)
    generator=torch.Generator(device=fields.device).manual_seed(seed)
    x=torch.randn((len(target),6,6),device=fields.device,generator=generator)
    indices=torch.linspace(99,0,steps,device=fields.device).round().long()
    for i,step in enumerate(indices):
        velocity=model.forward_cached(x,step.expand(len(target)),target,memory)
        alpha=schedule.alpha_bars[step]
        clean=alpha.sqrt()*x-(1-alpha).sqrt()*velocity
        noise=(1-alpha).sqrt()*x+alpha.sqrt()*velocity
        if i+1<len(indices):
            previous=schedule.alpha_bars[indices[i+1]]
            x=previous.sqrt()*clean+(1-previous).sqrt()*noise
        else:x=clean
    return model.denormalized(x,target)


def load_model(path,device='cuda'):
    payload=torch.load(path,map_location=device,weights_only=False)
    if payload.get('schema')!=SCHEMA or payload.get('channels')!=list(CHANNELS):raise ValueError('SDF checkpoint contract mismatch')
    model=SDFPoseDiT(**payload['architecture']).to(device);model.load_state_dict(payload['model']);model.eval()
    return model,payload
