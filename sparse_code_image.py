"""
Sparse code an image in 8x8 patches with ISTA or LCA, and save the reconstruction.

Usage:
    python sparse_code_image.py --method ista --image images/image.JPEG
    python sparse_code_image.py --method lca --image images/image.JPEG --max-side 0   (0 = full resolution)
"""
import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import ISTA
import LCA

PATCH = 8

def load_grayscale(path, max_side):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) #phone photos store rotation as metadata
    img = img.convert("L") #grayscale
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side)) #downscale, keeping aspect ratio
    return np.asarray(img, dtype=float) / 255.0 #pixel values in [0, 1]

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
    parser.add_argument("--method", choices=["ista", "lca"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-side", type=int, default=512, help="downscale so the longest side is at most this many pixels (0 = no downscaling)")
    args = parser.parse_args()

    img = load_grayscale(args.image, args.max_side)
    X, shape = to_patches(img)
    print(f"image {shape[1]}x{shape[0]} -> {X.shape[1]} patches")

    #remove each patch's mean brightness, then scale so values are ~unit variance like the random test images
    means = X.mean(axis=0)
    X = X - means
    scale = X.std() or 1.0
    X = X / scale

    if args.method == "ista":
        Phi = ISTA.Phi
        A = ISTA.ISTA(X, ISTA.lamby, Phi, ISTA.lr, Phi.shape[1])
    else:
        Phi = LCA.Phi
        A = LCA.LCA(X, LCA.lamby, Phi, LCA.lr, LCA.epsilon, Phi.shape[1])

    #undo the preprocessing and put the patches back together
    recon = from_patches((Phi @ A) * scale + means, shape)
    original = img[:shape[0], :shape[1]]

    mse = np.mean((recon - original) ** 2)
    print(f"MSE: {mse:.6f}   PSNR: {10 * np.log10(1 / mse):.2f} dB")
    print(f"active coefficients per patch: {np.count_nonzero(A, axis=0).mean():.1f} / {Phi.shape[1]}")

    os.makedirs("output", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    save(original, os.path.join("output", f"{name}_gray.png"))
    save(recon, os.path.join("output", f"{name}_{args.method}_recon.png"))
    print(f"saved output/{name}_gray.png and output/{name}_{args.method}_recon.png")
