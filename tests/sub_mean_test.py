
import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import torch 
def load_grayscale(path, max_side):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("L") 
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side)) 
    return np.asarray(img, dtype=float) / 255.0 

parser = argparse.ArgumentParser()


parser.add_argument("--image", required=True)
parser.add_argument("--max-side", type=int, default=512, help="downscale so the longest side is at most this many pixels (0 = no downscaling)")
args = parser.parse_args()
X = load_grayscale(args.image, args.max_side)
means = X.mean(axis=0)
X = X - means
scale = X.std() or 1.0
X = X / scale
recon = X
recon = np.clip((recon / np.max(abs(recon)))*0.5 + 0.5, min=0, max=1) * 255
def save(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)
save(recon, "sub_meaned.png")





