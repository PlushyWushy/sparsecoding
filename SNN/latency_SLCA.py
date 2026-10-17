import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so LCA.py in the folder above can be imported
import LCA

S = 500
cycles = 400
def spiking_LCA(input, lamby, Phi, lr, epsilon, S, T, num_bases, record=True):   # S = steps per cycle, T = cycle length in value units (largest representable x or a)
    Phi = torch.tensor(np.concatenate([Phi, -Phi], axis=1), dtype=torch.float32) #Doubled Dictionary
    x = torch.tensor(input, dtype=torch.float32)   
    x = torch.cat([torch.clamp(x, min=0), torch.clamp(-x, min=0)])   #On Off Split
    I = torch.eye(num_bases)
    feedforward = lr * torch.cat([Phi, -Phi]).T * epsilon  #On Off split
    recurrent = (I - lr * (Phi.T @ Phi)) * epsilon         
                               # IF threshold: v gains u - lamby*lr per step, so spikes per step = LCA's a
    u = torch.zeros_like(feedforward @ x) # same shape as the input current, so x can be one patch or a matrix of patches (one per column)
    u_prev = torch.zeros_like(feedforward @ x)   
    v = torch.zeros_like(u)
    dt = T / S
    theta = T + lr*lamby + dt / 2 #average amount over threshold is dt / 2, so we add to ensure no bias
    vs, spikes = [], []
    s = v >= theta
    t_r = torch.zeros_like(u) - 1
    start, n_out, n_in = time.perf_counter(), 0, 0   # wall-clock and spike counts, for comparing against spiking_LCA
  
    for c in range(cycles):
        u_prev = u
        u = (1-epsilon)*u
        """Encoding"""
        I = 0
        v = u_prev
        t_first = torch.full_like(u, S)   # step of each neuron's first spike this cycle; S = hasn't fired
        for t in range(S):
            v = v + dt              # IF neuron with starting membrane state u_prev
            s = v >= theta #spikes
            v = v - (s).float() * theta
            input_spikes_feedforward = (torch.clamp(torch.round(S - x/dt), min=0) == t).float()  
            input_spikes_recurrent = s.float()
            n_out, n_in = n_out + s.sum(), n_in + input_spikes_feedforward.sum()
            I += feedforward @ input_spikes_feedforward + recurrent @ input_spikes_recurrent
            u = u + I * dt
            if record:
                vs += [v]; spikes += [s]
            else:
                t_first = torch.where(s & (t_first == S), t, t_first)   # only track first spikes: storing every step for thousands of patches won't fit in memory
    print(f"latency: {S * cycles} steps (dt={dt:.3g}, {cycles * S * dt:.0f} time units), {n_out.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} recurrent spikes/patch, {n_in.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} input spikes/patch, {time.perf_counter() - start:.1f}s")
    if not record:
        return T - t_first * dt   # decode the final cycle; neurons that never fired get T - S*dt = 0
    spikes = torch.stack(spikes)
    final_cycle = spikes[-S:]
    diditspike = final_cycle.any(0).float()   # reduce over time (dim 0): one True/False per neuron (and per patch)
    a = ((T - final_cycle.int().argmax(0) * dt) * diditspike)
             
    return a, torch.stack(vs), spikes

if __name__ == "__main__":
    Phi = LCA.Phi
    image = LCA.image - LCA.image.mean()
    T = np.abs(image).max()   # cycle covers the largest ON/OFF input
    a, vs, spikes = spiking_LCA(image, LCA.lamby, Phi, LCA.lr, LCA.epsilon, S, T, 2 * Phi.shape[1])
    print(a)
    plt.imshow(spikes.T, aspect='auto', interpolation='nearest')
    plt.show()
