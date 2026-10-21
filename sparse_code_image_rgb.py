"""
Sparse code a color image in 8x8 patches with ISTA, LCA or latency-coded spiking LCA, one color channel at a time, and save the reconstruction.

Usage:
    python sparse_code_image_rgb.py --method ista --image images/image.JPEG
    python sparse_code_image_rgb.py --method lca --image images/image.JPEG --max-side 0   (0 = full resolution)
    python sparse_code_image_rgb.py --method latency_slca --image images/image.JPEG
"""
import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import classical.ISTA as ISTA
import classical.LCA as LCA
import SNN.config as config
from SNN import latency_SLCA as LSLCA
from sparse_code_image import to_patches, from_patches, save

def load_rgb(path, max_side):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) #phone photos store rotation as metadata
    img = img.convert("RGB") #drops alpha / converts grayscale or palette images to 3 channels
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side)) #downscale, keeping aspect ratio
    return np.asarray(img, dtype=float) / 255.0 #shape (h, w, 3), pixel values in [0, 1]

def code_channel(channel, method):
    X, shape = to_patches(channel)

    #remove each patch's mean brightness, then scale so values are ~unit variance like the random test images
    means = X.mean(axis=0)
    X = X - means
    scale = X.std() or 1.0
    X = X / scale

    if method == "ista":
        Phi = ISTA.Phi
        A = ISTA.ISTA(X, ISTA.lamby, Phi, ISTA.lr, Phi.shape[1])
    elif method == "lca":
        Phi = LCA.Phi
        A = LCA.LCA(X, LCA.lamby, Phi, LCA.lr, LCA.epsilon, Phi.shape[1])
    else:
        T = np.percentile(np.abs(X), 99.9)   # cycle length: covers 99.9% of ON/OFF inputs
        A = LSLCA.spiking_LCA(X, config.lamby, config.Phi, config.lr, config.epsilon, LSLCA.S, T, 2 * config.Phi.shape[1], record=False).numpy()
        Phi = np.concatenate([config.Phi, -config.Phi], axis=1)   # A has one coefficient per doubled-dictionary atom

    #undo the preprocessing and put the patches back together
    recon = from_patches((Phi @ A) * scale + means, shape)
    return recon, A

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ista", "lca", "latency_slca"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-side", type=int, default=512, help="downscale so the longest side is at most this many pixels (0 = no downscaling)")
    args = parser.parse_args()

    img = load_rgb(args.image, args.max_side)

    recon_channels = []
    for i, color in enumerate(["red", "green", "blue"]):
        recon, A = code_channel(img[:, :, i], args.method)
        recon_channels.append(recon)
        print(f"{color}: {A.shape[1]} patches, active coefficients per patch: {np.count_nonzero(A, axis=0).mean():.1f} / {A.shape[0]}")

    recon = np.stack(recon_channels, axis=2) #back to (h, w, 3)
    original = img[:recon.shape[0], :recon.shape[1]]

    mse = np.mean((recon - original) ** 2)
    print(f"MSE: {mse:.6f}   PSNR: {10 * np.log10(1 / mse):.2f} dB")

    os.makedirs("output", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    save(original, os.path.join("output", f"{name}_rgb.png"))
    save(recon, os.path.join("output", f"{name}_{args.method}_rgb_recon.png"))
    print(f"saved output/{name}_rgb.png and output/{name}_{args.method}_rgb_recon.png")
