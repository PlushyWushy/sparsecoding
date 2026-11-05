"""
Learn a dictionary Phi from natural image patches (Olshausen & Field 1996): alternate between
inferring sparse codes with Phi fixed (ISTA) and a gradient step on E = 1/2||x - Phi a||^2 + lambda||a||_1 with the codes fixed.

Usage:
    python SNN/phi_learner.py images/a.jpg images/b.jpg ...   (learns, shows the atoms)
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so classical.ISTA can be imported when running this file directly
import classical.ISTA as ISTA

def whitening_filter(shape, f_0=0.4):
    #see whitening_test.py
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    return f * np.exp(-(f / f_0) ** 4)

def whiten(img, f_0=0.4):
    return np.fft.ifft2(np.fft.fft2(img) * whitening_filter(img.shape, f_0)).real

def unwhiten(w, mean, f_0=0.4):
    R = whitening_filter(w.shape, f_0)
    R = (R > 0).astype(float) * R + (R <= 0).astype(float) * 100000000
    inverse = 1 / R
    return np.fft.ifft2(np.fft.fft2(w) * inverse).real + mean   #  add mean back

def sample_patches(paths, n, rng, patch=8, max_side=512, preprocess="sub-mean"):
    #same preprocessing as sparse_code_image.py, so Phi is learned on the statistics it will be used on
    imgs = []
    for p in paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("L")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        img = np.asarray(img, dtype=float) / 255.0
        imgs.append(whiten(img) if preprocess == "whiten" else img)
    X = np.empty((patch * patch, n))
    for i in range(n):   # random (overlapping) patch positions, spread evenly over the images
        img = imgs[rng.integers(len(imgs))]
        r, c = rng.integers(img.shape[0] - patch + 1), rng.integers(img.shape[1] - patch + 1)
        X[:, i] = img[r:r + patch, c:c + patch].ravel()
    if preprocess == "sub-mean":
        X = X - X.mean(axis=0)   # remove each patch's mean brightness
    return X / (X.std() or 1.0)   # ~unit variance

def learn_phi(X, num_bases, lamby, rng, iters=10000, batch=256, eta=10.0, steps=200):
    #X: one patch per column. Columns of Phi are kept at norm sqrt(patch dim), the expected norm of the random
    #Gaussian dictionary's columns, so lamby means the same thing it did before
    norm = np.sqrt(X.shape[0])
    Phi = rng.standard_normal((X.shape[0], num_bases))
    Phi *= norm / np.linalg.norm(Phi, axis=0) #initialize norm
    for it in range(iters):
        x = X[:, rng.integers(X.shape[1], size=batch)]
        lr = 1 / np.linalg.norm(Phi, 2) ** 2   # largest stable step for ISTA
        a = ISTA.ISTA(x, lamby, Phi, lr, num_bases, steps)
        residual = x - Phi @ a
        Phi += eta * residual @ a.T / batch   # -dE/dPhi, averaged over the batch
        Phi *= norm / np.maximum(np.linalg.norm(Phi, axis=0), 1e-12)   # otherwise Phi arbitrarily grows and a shrinks which is cheating
        if it % 100 == 0 or it == iters - 1:
            energy = np.mean(0.5 * np.sum(residual ** 2, axis=0) + lamby * np.sum(np.abs(a), axis=0))
            print(f"iter {it}: energy {energy:.4f}, active {np.count_nonzero(a, axis=0).mean():.1f} / {num_bases}")
    return Phi

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    if sys.argv[1:]:
        Phi = learn_phi(sample_patches(sys.argv[1:], 100000, rng), 128, 1.0, rng) 
        fig, axes = plt.subplots(8, 16, figsize=(16, 8))
        for k, ax in enumerate(axes.flat):
            ax.imshow(Phi[:, k].reshape(8, 8), cmap="gray"); ax.axis("off")
        plt.show()
    else:
        print('You need to run this on an image!')
