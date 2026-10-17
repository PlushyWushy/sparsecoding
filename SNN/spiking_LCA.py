import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so LCA.py in the folder above can be imported
import classical.LCA as LCA

T = 200000
def spiking_LCA(input, lamby, Phi, lr, epsilon, T, num_bases, record=True, dt=None):   # T = number of steps, dt = simulated time per step
    Phi = torch.tensor(np.concatenate([Phi, -Phi], axis=1), dtype=torch.float32) #Doubled Dictionary
    x = torch.tensor(input, dtype=torch.float32)   
    x = torch.cat([torch.clamp(x, min=0), torch.clamp(-x, min=0)])   #On Off Split
    I = torch.eye(num_bases)
    feedforward = lr * torch.cat([Phi, -Phi]).T * epsilon  #On Off split
    recurrent = (I - lr * (Phi.T @ Phi)) * epsilon         
    theta = 1.0                             
    dt = 1.0 / np.percentile(np.abs(input), 99.9)   # same range rule as latency_SLCA: the top 0.1% of ON/OFF cells saturate at one spike per step
    u = torch.zeros_like(feedforward @ x)   # same shape as the input current, so x can be one patch or a matrix of patches (one per column)
    v = torch.zeros_like(u)
    v_in = torch.zeros_like(x)              # membrane of the ON/OFF input cells

    vs, spikes = [], []
    count = 0
    s = v >= theta
    start, n_out, n_in = time.perf_counter(), 0, 0   # time and spike counts, for comparing against latency_SLCA
    for t in range(T):
        v_in = v_in + x * dt                   # IF neurons firing at rate x
        s_in = v_in >= 1.0
        v_in = v_in - s_in.float()
        u = recurrent @ s.float() + feedforward @ s_in.float() + (1 - epsilon * dt) * u   # leaky integrator: LCA.py's u, with spikes in place of a and x
        v = v + (u - lamby * lr) * dt          # IF neuron: integrate u - lambda
        s = v >= theta #spikes
        v = v - (s).float() * theta #reset
        n_out, n_in = n_out + s.sum(), n_in + s_in.sum()
        if record:
            vs += [v]; spikes += [s]
        elif t >= 1000:
            count = count + s.float()   # only count spikes: storing every step for thousands of patches won't fit in memory
    print(f"rate: {T} steps (dt={dt:.3g}, {T * dt:.0f} time units), {n_out.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} recurrent spikes/patch, {n_in.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} input spikes/patch, {time.perf_counter() - start:.1f}s")
    if not record:
        return count / ((T - 1000) * dt)   # spikes per unit time
    spikes = torch.stack(spikes)
    a = spikes[1000:].float().mean(0) / dt
    return a, torch.stack(vs), spikes

if __name__ == "__main__":
    Phi = LCA.Phi
    image = LCA.image - LCA.image.mean()
    a, vs, spikes = spiking_LCA(image, LCA.lamby, Phi, LCA.lr, LCA.epsilon, T, 2 * Phi.shape[1])
    print(a)
    plt.imshow(spikes.T, aspect='auto', interpolation='nearest')
    plt.show()
