"""
Sparse code an image in 8x8 patches with ISTA, LCA or spiking LCA (rate, latency or temporal pattern coded), and save the reconstruction.

Usage:
    python sparse_code_image.py --method ista --image images/image.JPEG
    python sparse_code_image.py --method lca --image images/image.JPEG --max-side 0   (0 = full resolution)
    python sparse_code_image.py --method spiking_lca --image images/image.JPEG
    python sparse_code_image.py --method latency_slca --image images/image.JPEG
    python sparse_code_image.py --method temporal_pattern_slca --image images/image.JPEG
    python sparse_code_image.py --method spiking_lca --image images/image.JPEG --preprocess whiten   
"""
import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import classical.ISTA as ISTA
import classical.LCA as LCA
import SNN.config as config
from SNN import spiking_LCA as SLCA
from SNN import latency_SLCA as LSLCA
from SNN import temporal_pattern_SLCA as TPSLCA
from SNN.phi_learner import whiten, unwhiten


PATCH = 8

def load_grayscale(path, max_side):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) #phone photos store rotation as metadata
    img = img.convert("L") #grayscale
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side)) #downscale, keeping aspect ratio
    return np.asarray(img, dtype=float) / 255.0 #pixel values in [0, 1] !!!! <-

def to_patches(img):
    h, w = img.shape
    h, w = h - h % PATCH, w - w % PATCH #crop so the image tiles evenly
    img = img[:h, :w]
    #(rows, 8, cols, 8) -> (rows, cols, 8, 8) -> one 64-long column per patch
    patches = img.reshape(h // PATCH, PATCH, w // PATCH, PATCH).swapaxes(1, 2).reshape(-1, PATCH * PATCH).T
    return patches, (h, w)

def from_patches(patches, shape):
    h, w = shape
    return patches.T.reshape(h // PATCH, w // PATCH, PATCH, PATCH).swapaxes(1, 2).reshape(h, w)

def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ista", "lca", "spiking_lca", "latency_slca", "temporal_pattern_slca", "mean"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-side", type=int, default=512, help="downscale so the longest side is at most this many pixels (0 = no downscaling)")
    parser.add_argument("--out-dir", default="output", help="results go in <out-dir>/<image name>/")
    parser.add_argument("--preprocess", choices=["whiten", "sub-mean"], default="sub-mean", help="preprocess the image either by whitening or subtracting the mean")
    args = parser.parse_args()

    img = load_grayscale(args.image, args.max_side)
    img = img[:img.shape[0] - img.shape[0] % PATCH, :img.shape[1] - img.shape[1] % PATCH]   # crop so the image tiles evenly (and whiten/unwhiten see the same size)
    original = img

    if args.preprocess == 'sub-mean':
        #remove each patch's mean brightness, then scale so values are ~unit variance like the random test images
        #(spiking_LCA then splits X into ON/OFF channels)
        X, shape = to_patches(img)
        means = X.mean(axis=0)
        X = X - means
        scale = X.std() or 1.0
        X = X / scale
    elif args.preprocess == 'whiten':
        #apply whitening, then cut into patches and scale to ~unit variance
        mean = img.mean()   # whitening erases the mean brightness, so keep it to add back
        X, shape = to_patches(whiten(img))
        scale = X.std() or 1.0
        X = X / scale
    print(f"image {shape[1]}x{shape[0]} -> {X.shape[1]} patches")

    if args.method == "ista":
        Phi = ISTA.Phi
        A = ISTA.ISTA(X, ISTA.lamby, Phi, ISTA.lr, Phi.shape[1])
        X_hat = Phi @ A
    elif args.method == "lca":
        Phi = LCA.Phi
        A = LCA.LCA(X, LCA.lamby, Phi, LCA.lr, LCA.epsilon, Phi.shape[1])
        X_hat = Phi @ A
    elif args.method == "spiking_lca":
        Phi, lr = config.load_phi(args.preprocess)
        A = SLCA.spiking_LCA(X, config.lamby, Phi, lr, config.epsilon, SLCA.T, 2 * Phi.shape[1], record=False).numpy()
        X_hat = np.concatenate([Phi, -Phi], axis=1) @ A
    elif args.method == "latency_slca":
        Phi, lr = config.load_phi(args.preprocess)
        T = np.abs(X).max()   # cycle length: covers the largest ON/OFF input
        A = LSLCA.spiking_LCA(X, config.lamby, Phi, lr, config.epsilon, LSLCA.S, T, 2 * Phi.shape[1], record=False).numpy()
        X_hat = np.concatenate([Phi, -Phi], axis=1) @ A
    elif args.method == 'temporal_pattern_slca':
        Phi, lr = config.load_phi(args.preprocess)
        T = np.abs(X).max()   # representable range: covers the largest ON/OFF input
        A = TPSLCA.spiking_LCA(X, config.lamby, Phi, lr, config.epsilon, TPSLCA.S, T, 2 * Phi.shape[1], record=False).numpy()
        X_hat = np.concatenate([Phi, -Phi], axis=1) @ A
    elif args.method == "mean":   # control: all-zero code, so only what preprocessing removed comes back (each patch's mean for sub-mean, the image's mean for whiten)
        Phi, lr = config.load_phi(args.preprocess)
        A = np.zeros((Phi.shape[1], X.shape[1]))
        X_hat = Phi @ A
    else:
        print('You ought to choose a method!')

    #undo the preprocessing and put the patches back together
    if args.preprocess == 'sub-mean':
        recon = from_patches(X_hat * scale + means, shape)
    elif args.preprocess == 'whiten':
        recon = unwhiten(from_patches(X_hat * scale, shape), mean)


    mse = np.mean((recon - original) ** 2)
    print(f"MSE: {mse:.6f}   PSNR: {10 * np.log10(1 / mse):.2f} dB")
    print(f"active coefficients per patch: {np.count_nonzero(A, axis=0).mean():.1f} / {Phi.shape[1]}")

    name = os.path.splitext(os.path.basename(args.image))[0]
    out = os.path.join(args.out_dir, name)   # one folder per image, holding the original and every method's reconstruction
    os.makedirs(out, exist_ok=True)
    save(original, os.path.join(out, "gray.png"))
    save(recon, os.path.join(out, f"{args.method}_recon.png"))
    print(f"saved {out}/gray.png and {out}/{args.method}_recon.png")
