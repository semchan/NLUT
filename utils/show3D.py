import numpy as np
from scipy.spatial import distance
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import pdb
import os
from os.path import join
import sys
sys.path.append("./utils")
from LUT import *

root = "/mnt/tvmn1/input"
name = "Test+10-1-1_models/"
epoch = 335

path_test = "/home/chenys/surface/NLUT_NET/finetuning_train/checkpoint/0_finetuning_style_lut.pth"
model = torch.load((path_test),map_location=torch.device('cpu'))
luts_model = model['state_dict']["CLUTs.LUTs"]
luts = cube_to_lut(luts_model.reshape(-1,3,32,32,32))

# model = torch.load(os.path.join(root, name, "model{:0>4}.pth".format(epoch)),map_location=torch.device('cpu'))
# luts = cube_to_lut(model['LUT_model.LUTs'].reshape(-1,3,33,33,33))
# idt = identity3d_tensor(33)
idt = identity3d_tensor(32)
luts += idt.unsqueeze(0)


# dim = 33    
dim = 32 
x, y, z = np.arange(0,dim), np.arange(0,dim), np.arange(0,dim)
xx, yy, zz = np.meshgrid(x, y, z)
coords = np.array((xx.ravel(), yy.ravel(), zz.ravel())).T




fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])
fig.add_trace(
    go.Scatter3d(x=coords[:, 2],
                 y=coords[:, 0],
                 z=coords[:, 1],
                 mode='markers',
                 marker=dict(
                    color=idt.reshape(3,-1).transpose(0,1),
                    opacity=0.5,
                 )),
    row=1, col=1
)
lut = luts[0].reshape(3,-1).transpose(0,1)
fig.add_trace(
    go.Scatter3d(x=lut[:, 0],
                 y=lut[:, 1],
                 z=lut[:, 2],
                 mode='markers',
                 marker=dict(
                    color=idt.reshape(3,-1).transpose(0,1),
                    opacity=0.5,
                 )),
    row=1, col=2
)



# fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])
# fig.add_trace(
#     go.Scatter3d(x=coords[:, 2],
#                  y=coords[:, 0],
#                  z=coords[:, 1],
#                  mode='markers',
#                  marker=dict(
#                     color=idt.reshape(3,-1).transpose(0,1),
#                     opacity=0.5,
#                  )),
#     row=1, col=1
# )
# lut = luts[4].reshape(3,-1).transpose(0,1)
# fig.add_trace(
#     go.Scatter3d(x=lut[:, 0],
#                  y=lut[:, 1],
#                  z=lut[:, 2],
#                  mode='markers',
#                  marker=dict(
#                     color=idt.reshape(3,-1).transpose(0,1),
#                     opacity=0.5,
#                  )),
#     row=1, col=2
# )

fig.write_image('spx00.png')

print("over")