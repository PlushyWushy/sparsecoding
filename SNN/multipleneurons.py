import torch
import matplotlib.pyplot as plt

dt, tau, theta, T = 1e-3, 2e-2, 1.0, 500
beta = torch.exp(torch.tensor(-dt / tau))
I = torch.tensor([1.1, 4.2, 2, 2.3])                                 # constant input, same units as v

v = torch.tensor([0.0, 0.0, 0.0, 0.0])


vs, spikes = [], []
for t in range(T):
    v = ((beta) * v + I*(1 - beta))
    not_s = v < theta 
    s = v >= theta
    v = v * (not_s).float()
 
    vs += [v]; spikes += [s]
vs, spikes = torch.stack(vs), torch.stack(spikes)
fig, axes = plt.subplots(len(v), 1, sharex=True)
for k in range(len(axes)):
    axes[k].plot(vs[:, k])
plt.show()




