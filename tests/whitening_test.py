
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
F = np.fft.fft2(X)
f_0 = 0.4
print(F)
fy = np.fft.fftfreq(F.shape[0]) 
fx = np.fft.fftfreq(F.shape[1]) 
for i in range(len(F)):
    for j in range(len(F[i])):
        f = np.sqrt(fy[i]**2 + fx[j]**2)
        F[i][j] = F[i][j] * f*np.exp(-((f/f_0)**4))
print(F)
recon = np.fft.ifft2(F).real
recon = np.clip((recon / np.max(abs(recon)))*0.5 + 0.5, min=0, max=1) * 255
def save(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)
save(recon, "tests/whitened.png")





