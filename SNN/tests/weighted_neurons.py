import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)
dt, tau, theta, T = 1e-3, 2e-2, 1.0, 500
beta = torch.exp(torch.tensor(-dt / tau))                             
v = torch.zeros(8)
feedforward = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
recurrent = torch.randn((8, 8))
x = torch.tensor([3.14, 2.718])

vs, spikes = [], []
s = v >= theta
for t in range(T):
    v = ((beta) * v + (feedforward @ x + recurrent @ s.float())*(1 - beta))
    not_s = v < theta 
    s = v >= theta
    v = v * (not_s).float()
    vs += [v]; spikes += [s]
plt.imshow(spikes, aspect='auto')
plt.show()