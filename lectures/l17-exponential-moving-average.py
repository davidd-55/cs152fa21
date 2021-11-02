# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")

# %%
x = torch.linspace(0, 10, 100)
y = torch.sin(x) + torch.randn_like(x) * 0.5

# %%
plt.plot(x, y)

# %%
y_ema = [y[0]]
beta = 0.8

for yval in y[1:]:
    new_y_ema = beta * y_ema[-1] + (1 - beta) * yval
    y_ema.append(new_y_ema)

plt.plot(x, torch.sin(x))
plt.plot(x, y)
plt.plot(x, y_ema)

# %%
