"""
Complex ISTA with slowness penalty
TODO:
Make it actually the cadieu olshausen formulation
"""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence


lr = 0.001
lamby = 0.4
def gabor(idx, sigma=1.5, period=4):
    return np.exp(-idx**2 / (2 * sigma**2)) * np.exp(2j * np.pi * idx / period) #complex version of RingLCA's wavelet; real part is the cosine gabor, imaginary part the sine
Phi = gabor(np.arange(64)[:, None] - (np.arange(128)/2)[None, :]) #same layout as RingLCA: 128 centers at half-pixel steps along the 64 (flattened) pixels
beta = 0.1
def load_movie(path):
    video = np.stack([np.asarray(f.convert("L"), dtype=float).flatten() / 255 for f in ImageSequence.Iterator(Image.open(path))]) #(num_frames, pixels)
    if video.shape[1] != Phi.shape[0]:
        raise ValueError(f"{path} has {video.shape[1]} pixels per frame, but Phi expects {Phi.shape[0]} (8x8)")
    return video
def frame_changes(a, a_prev):
    d_amp = np.abs(a) - np.abs(a_prev)
    d_phase = np.angle(a * a_prev.conj()) #wrapped to (-pi, pi]
    d_phase[(a == 0) | (a_prev == 0)] = np.nan #phase is undefined for inactive coefficients
    return d_amp, d_phase
def ISTA(input, lamby, Phi, beta, lr, num_bases, a_prev):
    x = input
    I = np.eye(num_bases)
    a = np.zeros(num_bases)
    recurrent1 = (I - lr * ((Phi.conj().transpose() @ Phi)) - lr*2*beta * I)
    recurrent2 = 0.5 * lr * Phi.conj().transpose() @ Phi.conj()
    feedforward = lr * Phi.conj().transpose() @ x
    for i in range(1000):
        z = recurrent1 @ a + feedforward - recurrent2 @ a.conj() + lr * 2 * beta * np.abs(a_prev) * np.sign(a)  #slowness pulls |a| toward |a_t-1|
        a = np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0) #np.sign(z) returns z/|z| = a/|a|
    return a
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("movie", help="path to a movie (e.g. movies/ball.gif) with 8x8 grayscale frames")
    args = parser.parse_args()

    video = load_movie(args.movie)
    num_frames, num_bases = video.shape[0], Phi.shape[1]
    A = np.zeros((num_frames, num_bases), dtype=complex)
    d_amp = np.full((num_frames - 1, num_bases), np.nan) #row t is the change from frame t to t+1
    d_phase = np.full((num_frames - 1, num_bases), np.nan)
    a_prev = np.zeros(num_bases, dtype=complex)
    for t, frame in enumerate(video):
        a = ISTA(frame, lamby, Phi, beta, lr, num_bases, a_prev)
        A[t] = a
        err = np.linalg.norm(frame - np.real(Phi @ a)) / (np.linalg.norm(frame) or 1.0)
        line = f"frame {t:2d}: rel err {err:.3f}, active {np.count_nonzero(a):3d}"
        if t > 0:
            d_amp[t - 1], d_phase[t - 1] = frame_changes(a, a_prev)
            both = (a != 0) & (a_prev != 0) #percent change only makes sense for coefficients active in both frames
            pct_amp = np.mean(100 * np.abs(d_amp[t - 1][both]) / np.abs(a_prev[both])) if both.any() else np.nan
            line += f", mean |d amp| {np.mean(np.abs(d_amp[t - 1])):.4f} ({pct_amp:.0f}% of |a| for {both.sum()} active in both), mean |d phase| {np.nanmean(np.abs(d_phase[t - 1])):.3f} rad"
        print(line)
        a_prev = a

    name = Path(args.movie).stem
    os.makedirs(os.path.join("output", name), exist_ok=True)
    out = os.path.join("output", name, "complex_ista.npz")
    np.savez(out, A=A, d_amp=d_amp, d_phase=d_phase)
    print(f"saved {out} (A: coefficients per frame, d_amp/d_phase: change between consecutive frames)")
