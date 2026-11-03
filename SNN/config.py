import glob
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent)) 
from SNN.phi_learner import sample_patches, learn_phi
sys.path.append(str(Path(__file__).resolve().parent.parent))
# parameters shared by the spiking LCAs (rate, latency, temporal pattern); start from classical/LCA.py's values
lamby = 1.0
epsilon = 0.1
num_bases = 128
dataset = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "images" / "kodak" / "*.png")))   
rng = np.random.default_rng(0)

# learning takes a few minutes, so the learned Phi is cached and only relearned when the settings above change
cache = Path(__file__).with_name("phi_learned.npz")
key = repr((dataset, num_bases, lamby))
if cache.exists() and np.load(cache)["key"] == key:
    Phi = np.load(cache)["Phi"]
else:
    Phi = learn_phi(sample_patches(dataset, 100000, rng), num_bases, lamby, rng)
    np.savez(cache, Phi=Phi, key=key)

lr = 1 / np.linalg.norm(Phi, 2) ** 2   # largest stable step; this is where 0.0027 came from for the old random Gaussian Phi
image = rng.standard_normal(64)
if __name__ == '__main__':
    print(Phi)
    if sys.argv[1]:
        plt.plot(Phi[int(sys.argv[1])]); plt.show()
    else:
        plt.plot(Phi); plt.show()
