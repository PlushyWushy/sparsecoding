"""
Makes toy movies for testing the complex (slowness-penalized) sparse coders
"""
from pathlib import Path
import numpy as np
from PIL import Image


def ball_video(size=8):
    # 2x2 block of white pixels moving left to right one pixel per frame, shape (size - 1, size, size) in {0, 1}
    top = size // 2 - 1
    frames = np.zeros((size - 1, size, size))
    for t in range(size - 1):
        frames[t, top:top + 2, t:t + 2] = 1
    return frames


if __name__ == "__main__":
    frames = [Image.fromarray(np.uint8(np.round(f * 255)), mode="L") for f in ball_video()]
    out = Path(__file__).resolve().parent / "ball.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)
    print(f"saved {out}")
