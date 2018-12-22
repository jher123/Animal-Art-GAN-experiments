import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt

# Plotting

# from fast.ai notebook
def gallery(x, nc=3):
    n,h,w,c = x.shape
    nr = n//nc
    assert n == nr*nc
    return (x.reshape(nr, nc, h, w, c)
              .swapaxes(1,2)
              .reshape(h*nr, w*nc, c))


# def plot_debug_info(_debug_info_dict):
#     plt.figure(figsize=(15, 15))
#     fig, axs = plt.subplots(2, 2)
#     pd.Series([item[0] for item in _debug_info_dict['lossG']]).plot(ax=axs[0, 0], title='lossG')
#     pd.Series([item[0] for item in _debug_info_dict['fake_res']]).plot(ax=axs[0, 1], title='D(G(z))')
#     pd.Series([item[0] for item in _debug_info_dict['lossD']]).plot(ax=axs[1, 0], title='lossD' )
#     pd.Series([item[0] for item in _debug_info_dict['real_res']]).plot(ax=axs[1, 1], title='D(x)')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def set_trainable(model, flag):
    model.trainable = flag
    for p in model.paramaters():
        p.requires_grad = flag
