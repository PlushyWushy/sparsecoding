import glob
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from SNN.phi_learner import sample_patches, learn_phi
# parameters shared by the spiking LCAs (rate, latency, temporal pattern); start from classical/LCA.py's values
lamby = 0.1
epsilon = 0.1
num_bases = 128
dataset = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "images" / "kodak" / "*.png")))
rng = np.random.default_rng(0)

def load_phi(preprocess="whiten"):
    if preprocess == "sub-mean":
        cache = Path(__file__).with_name("phi_learned.npz")
        key = repr((dataset, num_bases, lamby))
    else:
        cache = Path(__file__).with_name(f"phi_learned_{preprocess}.npz")
        key = repr((dataset, num_bases, lamby, preprocess))
    if cache.exists() and np.load(cache)["key"] == key:
        Phi = np.load(cache)["Phi"]
    else:
        rng = np.random.default_rng(0)
        Phi = learn_phi(sample_patches(dataset, 100000, rng, preprocess=preprocess), num_bases, lamby, rng)
        np.savez(cache, Phi=Phi, key=key)
    return Phi, 1 / np.linalg.norm(Phi, 2) ** 2   # largest stable step; this is where 0.0027 came from for the old random Gaussian Phi

Phi, lr = load_phi("whiten")
image = rng.standard_normal(64)
if __name__ == '__main__':
    print(Phi)
    try:
        #plt.plot(Phi[:, int(sys.argv[1])]); plt.show()
        plt.imshow(Phi[:, int(sys.argv[1])].reshape(8, 8), cmap="gray"); plt.show()
    except:
        plt.plot(Phi); plt.show()
