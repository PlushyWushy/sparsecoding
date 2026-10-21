import numpy as np

# parameters shared by the spiking LCAs (rate, latency, temporal pattern); start from classical/LCA.py's values
lr = 0.0027
lamby = 4.0
epsilon = 0.1
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
image = rng.standard_normal(64)
