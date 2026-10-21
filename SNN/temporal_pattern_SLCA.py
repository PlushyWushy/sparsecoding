import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so SNN.config can be imported when running this file directly
import SNN.config as config

S = 9 #2^9 = 500 representable values
cycles = 400
def spiking_LCA(input, lamby, Phi, lr, epsilon, S, T, num_bases, record=True):   # S = steps per cycle (bits), T = representable range in value units (largest representable x or a)
    Phi = torch.tensor(np.concatenate([Phi, -Phi], axis=1), dtype=torch.float32) #Doubled Dictionary
    x = torch.tensor(input, dtype=torch.float32)   
    x = torch.cat([torch.clamp(x, min=0), torch.clamp(-x, min=0)])   #On Off Split
    I = torch.eye(num_bases)
    feedforward = lr * torch.cat([Phi, -Phi]).T * epsilon  #On Off split
    recurrent = (I - lr * (Phi.T @ Phi)) * epsilon         
                            
    u = torch.zeros_like(feedforward @ x) # same shape as the input current
    u_prev = torch.zeros_like(feedforward @ x)   
    v = torch.zeros_like(u)
    q = T / 2**S   # value of the smallest bit (last step of the cycle)
    theta = T / 2 ** torch.arange(1, S + 1, dtype=torch.float32)   # threshold halves every step; theta_t = q * 2^(S-t-1)
    vs, spikes = [], []
    s = v >= theta[0]
    start, n_out, n_in = time.perf_counter(), 0, 0   # wall-clock and spike counts
  
    for c in range(cycles):
        u_prev = u
        u = (1-epsilon)*u
        """Encoding"""
        v = u_prev - lr*lamby + q / 2   # IF neuron starts at u - lambda, no input current; + q/2 rounds to nearest instead of flooring (no bias)
        a_last = torch.zeros_like(u)   # this cycle's decoded output
        for t in range(S):
            s = v >= theta[t] #spikes: bit t of (u - lambda)
            v = v - (s).float() * theta[t]   # subtractive reset by the current (halving) threshold
            input_spikes_feedforward = (torch.floor(torch.clamp(x/q + 0.5, max=2**S - 1) / 2**(S-1-t)) % 2)   # bit t of x
            input_spikes_recurrent = s.float()
            n_out, n_in = n_out + s.sum(), n_in + input_spikes_feedforward.sum()
            I = feedforward @ input_spikes_feedforward + recurrent @ input_spikes_recurrent
            u = u + I * theta[t]   # a spike at step t is worth 2^(S-t-1) bits, so u steps up by w * theta_t
            if record:
                vs += [v]; spikes += [s]
            else:
                a_last = a_last + s.float() * theta[t]   
    print(f"temporal pattern: {S * cycles} steps (q={q:.3g}), {n_out.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} recurrent spikes/patch, {n_in.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} input spikes/patch, {time.perf_counter() - start:.1f}s")
    if not record:
        return a_last   # decoded final cycle; neurons that never fired get 0
    spikes = torch.stack(spikes)
    final_cycle = spikes[-S:]
    a = (final_cycle.float() * theta.reshape(-1, *[1] * (final_cycle.dim() - 1))).sum(0)   # weight each step's spike by its bit value and sum over time (dim 0)
             
    return a, torch.stack(vs), spikes

if __name__ == "__main__":
    Phi = config.Phi
    image = config.image - config.image.mean()
    T = np.abs(image).max()   # cycle covers the largest ON/OFF input
    a, vs, spikes = spiking_LCA(image, config.lamby, Phi, config.lr, config.epsilon, S, T, 2 * Phi.shape[1])
    print(a)
    plt.imshow(spikes.T, aspect='auto', interpolation='nearest')
    plt.show()
