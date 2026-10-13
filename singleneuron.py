import torch
import matplotlib.pyplot as plt

dt, tau, theta, T = 1e-3, 2e-2, 1.0, 5000
beta = torch.exp(torch.tensor(-dt / tau))
I = 0.0                                  # constant input, same units as v

v = torch.tensor(0.0)
sweep = []
for i in range(50):
    vs, spikes = [], []
    for t in range(T):
        v = ((beta) * v + I*(1 - beta)) 
        if v > theta:
            s = torch.tensor(1.0)
            v = torch.tensor(0.0)
        else:
            s = torch.tensor(0.0)   
        vs.append(v.item()); spikes.append(s.item())
        sweep.append(spikes.count(1.0) / 5000)
    I += 0.1
plt.plot(sweep); plt.show()

