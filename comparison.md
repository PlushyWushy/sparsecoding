# Spiking LCA comparison on Kodak

Rate-coded (`spiking_lca`), latency-coded (`latency_slca`) and temporal-pattern-coded (`temporal_pattern_slca`) spiking LCA, run on all 24 Kodak images. The control (`mean`) reconstructs each patch from its mean brightness only, i.e. an all-zero code.

## Setup

| | |
|---|---|
| Images | Kodak `kodim01`–`kodim24`, grayscale, downscaled to 512 px on the long side (512×336 or 336×512, 2688 patches each) |
| Patches | 8×8, patch mean removed, scaled to unit variance per image |
| Dictionary | `SNN/phi_learned.npz`: 64×128, learned on 100k random Kodak patches, columns at norm 8 |
| Parameters | λ = 1.0, ε = 0.1, lr = 1/‖Φ‖₂² = 0.00148 (`SNN/config.py`) |
| Rate | 204,800 steps, dt = 1 / (99.9th percentile of \|x\|) per image |
| Latency | 400 cycles × 512 steps = 204,800 steps, T = 99.9th percentile of \|x\| |
| Temporal pattern | 400 cycles × 9 bits = 3,600 steps, same T, so q = T/512, the same grid as latency |

Coded images are decoded from the final state. "Active" is the number of nonzero coefficients per patch in the doubled (±Φ) dictionary.

## Summary (mean ± std over 24 images)

| Method | PSNR (dB) | Gain over control (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch | Steps | Wall-clock / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control (mean only) | 22.67 ± 2.44 | — | 0 | — | — | — | — |
| Rate | **43.40 ± 2.33** | **20.73 ± 0.92** | 38.7 ± 8.0 | 65,068 | 1,338,630 | 204,800 | 731 s |
| Latency | 35.45 ± 2.40 | 12.78 ± 1.23 | 31.0 ± 10.2 | **12,215** | **24,952** | 204,800 | 734 s |
| Temporal pattern | 35.45 ± 2.40 | 12.78 ± 1.22 | 31.0 ± 10.3 | 18,098 | 72,266 | **3,600** | **10.7 s** |

Paired PSNR differences (same image, 95% CI):

| Comparison | Mean difference | Images where the first is better |
|---|---:|---:|
| Temporal pattern − latency | +0.002 ± 0.007 dB | 11 / 24 |
| Rate − latency | +7.95 ± 0.25 dB | 24 / 24 |
| Rate − temporal pattern | +7.95 ± 0.25 dB | 24 / 24 |

Wall-clock was measured with 6 jobs running at once, 3 threads each.

## Per-image PSNR (dB) and active coefficients per patch

| Image | Control | Rate | Latency | TP | Active (rate) | Active (lat / TP) |
|---|---:|---:|---:|---:|---:|---:|
| kodim01 | 20.64 | 41.02 | 33.06 | 33.06 | 48.7 | 46.0 / 46.0 |
| kodim02 | 27.55 | 45.50 | 37.68 | 37.69 | 43.8 | 20.0 / 20.0 |
| kodim03 | 25.65 | 46.77 | 38.89 | 38.89 | 30.9 | 21.2 / 21.2 |
| kodim04 | 25.02 | 44.35 | 36.92 | 36.91 | 40.8 | 28.9 / 28.8 |
| kodim05 | 18.49 | 39.31 | 31.32 | 31.30 | 46.2 | 45.3 / 45.2 |
| kodim06 | 22.01 | 42.78 | 34.07 | 34.10 | 42.4 | 32.9 / 33.0 |
| kodim07 | 22.44 | 44.52 | 37.81 | 37.83 | 30.6 | 34.4 / 34.5 |
| kodim08 | 16.92 | 38.20 | 31.61 | 31.62 | 40.3 | 44.5 / 44.6 |
| kodim09 | 23.44 | 45.07 | 37.55 | 37.52 | 28.8 | 20.6 / 20.5 |
| kodim10 | 24.14 | 45.07 | 37.42 | 37.44 | 33.3 | 24.6 / 24.7 |
| kodim11 | 22.66 | 43.47 | 35.11 | 35.14 | 39.7 | 29.7 / 29.8 |
| kodim12 | 23.97 | 45.07 | 37.10 | 37.09 | 32.7 | 17.7 / 17.7 |
| kodim13 | 19.56 | 39.22 | 30.87 | 30.87 | 55.2 | 51.9 / 52.0 |
| kodim14 | 21.55 | 41.92 | 33.84 | 33.85 | 48.3 | 42.7 / 42.7 |
| kodim15 | 21.88 | 43.70 | 35.73 | 35.75 | 28.0 | 17.0 / 17.0 |
| kodim16 | 26.10 | 46.66 | 38.16 | 38.13 | 43.2 | 35.3 / 35.2 |
| kodim17 | 23.49 | 43.89 | 36.80 | 36.80 | 37.2 | 32.5 / 32.6 |
| kodim18 | 21.83 | 41.89 | 33.00 | 33.03 | 47.1 | 37.2 / 37.3 |
| kodim19 | 21.38 | 42.57 | 34.88 | 34.89 | 36.0 | 27.0 / 27.1 |
| kodim20 | 22.44 | 44.76 | 36.41 | 36.39 | 23.5 | 16.5 / 16.5 |
| kodim21 | 21.77 | 42.66 | 34.57 | 34.56 | 35.5 | 30.2 / 30.1 |
| kodim22 | 24.41 | 44.42 | 35.36 | 35.37 | 46.1 | 31.4 / 31.4 |
| kodim23 | 25.39 | 46.60 | 39.07 | 39.06 | 29.0 | 19.4 / 19.4 |
| kodim24 | 21.31 | 42.14 | 33.56 | 33.54 | 42.2 | 37.8 / 37.7 |

## Files and reproduction

- `output/kodak/kodimXX/` contains `gray.png` (the original) and, for each method, `<method>_recon.png` and `<method>.log` (all printed output: spike counts, MSE, PSNR, active).
- `output/kodak/results.csv` has every number above, one row per image and method.

To regenerate a single result:

```
python sparse_code_image.py --method <mean|spiking_lca|latency_slca|temporal_pattern_slca> \
    --image images/kodak/kodimXX.png --out-dir output/kodak > output/kodak/kodimXX/<method>.log
```
