import torch
import matplotlib.pyplot as plt

dt, tau, theta, T = 1e-3, 2e-2, 1.0, 500
beta = torch.exp(torch.tensor(-dt / tau))
I = torch.linspace(0, 5, 50)                              
v = torch.linspace(0, 0, 50)

vs, spikes = [], []
for t in range(T):
    v = ((beta) * v + I*(1 - beta))
    not_s = v < theta 
    s = v >= theta
    v = v * (not_s).float()
 
    vs += [v]; spikes += [s]
plt.plot(I, sum(spikes) / T)
plt.show()




