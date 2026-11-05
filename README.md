# Sparse Coding for Spiking Neural networks

See [`Alternative_Coding_Methods.pdf`](Alternative_Coding_Methods.pdf) for formulations.

To visualize Phi, run ```python SNN/config.py n``` for an index n in Phi, or just run ```python SNN/config.py``` to see all of it.
---

# Spiking LCA comparison on Kodak

Rate-coded (`spiking_lca`), latency-coded (`latency_slca`) and temporal-pattern-coded (`temporal_pattern_slca`) spiking LCA, run on all 24 Kodak images with whitening preprocessing (`--preprocess whiten`). The control (`mean`) is an all-zero code, so it reconstructs every pixel as the image's mean brightness.

## Setup

| | |
|---|---|
| Images | Kodak `kodim01`–`kodim24`, grayscale, downscaled to 512 px on the long side (512×336 or 336×512, 2688 patches each) |
| Whitening | Whole image filtered in Fourier space by R(f) = f·exp(−(f/0.4)⁴), f = √(fx² + fy²), then cut into 8×8 patches and scaled to unit variance per image |
| Unwhitening | Multiply by 1/R at every f > 0 (f = 0 set to 0), undo the scale, add the image mean back |
| Dictionary | `SNN/phi_learned_whiten.npz`: 64×128, learned on 100k random patches of whitened Kodak images, columns at norm 8 |
| Parameters | λ = 1.0, ε = 0.1, lr = 1/‖Φ‖₂² = 0.00257 (`SNN/config.py`) |
| Rate | 204,800 steps, dt = 1 / max\|x\| per image |
| Latency | 400 cycles × 512 steps = 204,800 steps, T = max\|x\| per image, leftover v carried across cycles (see Rounding error) |
| Temporal pattern | 400 cycles × 9 bits = 3,600 steps, same T, so q = T/512, the same grid as latency, leftover v carried across cycles |

PSNR = 10·log10(1/MSE) on pixels in [0, 1]. "Active" is the number of nonzero coefficients per patch in the doubled (±Φ) dictionary.

## Summary (mean ± std over 24 images)

| Method | PSNR (dB) | Difference from control (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch | Steps | Wall-clock / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control (mean only) | 15.12 ± 2.47 | — | 0 | — | — | — | — |
| Rate | 14.12 ± 3.01 | −1.00 ± 3.14 | 40.1 ± 6.1 | 44,797 | 860,211 | 204,800 | 762 s |
| Latency | 14.49 ± 2.85 | −0.62 ± 3.06 | 34.6 ± 8.0 | 16,120 | 25,011 | 204,800 | 847 s |
| Temporal pattern | 14.55 ± 2.87 | −0.56 ± 3.13 | 34.6 ± 7.9 | 21,474 | 65,999 | 3,600 | 12.7 s |

Wall-clock was measured with 6 jobs running at once, 3 threads each. Earlier results are in `legacy_tests/`. 

## Per-image PSNR (dB) and active coefficients per patch

| Image | Control | Rate | Latency | TP | Active (rate) | Active (lat / TP) |
|---|---:|---:|---:|---:|---:|---:|
| kodim01 | 16.52 | 10.93 | 11.80 | 11.77 | 46.9 | 46.6 / 46.7 |
| kodim02 | 22.24 | 17.55 | 17.48 | 17.17 | 44.2 | 31.3 / 31.3 |
| kodim03 | 16.31 | 18.18 | 18.72 | 18.88 | 35.2 | 27.7 / 27.5 |
| kodim04 | 16.67 | 17.29 | 17.33 | 17.44 | 42.6 | 34.2 / 33.9 |
| kodim05 | 14.67 | 9.26 | 10.11 | 10.04 | 46.3 | 43.9 / 43.8 |
| kodim06 | 12.70 | 12.47 | 13.23 | 13.17 | 42.8 | 38.8 / 38.9 |
| kodim07 | 16.89 | 15.21 | 16.02 | 16.15 | 33.8 | 29.9 / 29.8 |
| kodim08 | 12.44 | 8.36 | 9.08 | 9.01 | 41.9 | 41.3 / 41.2 |
| kodim09 | 16.68 | 17.62 | 16.96 | 17.01 | 31.3 | 22.9 / 23.3 |
| kodim10 | 16.43 | 15.68 | 16.15 | 16.21 | 36.5 | 26.3 / 26.5 |
| kodim11 | 15.92 | 13.86 | 13.99 | 14.17 | 41.4 | 35.8 / 36.0 |
| kodim12 | 15.04 | 15.35 | 16.96 | 17.08 | 37.0 | 28.8 / 28.9 |
| kodim13 | 14.09 | 8.46 | 8.83 | 8.94 | 51.7 | 52.5 / 52.0 |
| kodim14 | 14.72 | 12.89 | 13.17 | 13.22 | 48.1 | 44.6 / 44.7 |
| kodim15 | 9.85 | 15.51 | 16.21 | 16.16 | 33.9 | 27.4 / 27.5 |
| kodim16 | 15.63 | 17.42 | 17.40 | 17.54 | 43.9 | 39.3 / 39.1 |
| kodim17 | 14.47 | 15.08 | 15.32 | 15.51 | 39.8 | 31.7 / 31.8 |
| kodim18 | 16.70 | 11.53 | 12.14 | 12.38 | 46.2 | 41.4 / 41.1 |
| kodim19 | 14.90 | 13.45 | 13.60 | 13.53 | 36.6 | 29.9 / 29.9 |
| kodim20 | 9.38 | 13.78 | 13.79 | 14.25 | 27.4 | 23.2 / 23.2 |
| kodim21 | 15.86 | 13.43 | 13.90 | 13.93 | 37.0 | 33.1 / 32.9 |
| kodim22 | 16.09 | 16.07 | 16.01 | 15.92 | 45.8 | 39.3 / 39.2 |
| kodim23 | 14.84 | 18.69 | 18.64 | 18.88 | 30.6 | 21.2 / 21.4 |
| kodim24 | 13.74 | 10.69 | 11.01 | 10.92 | 42.2 | 39.5 / 39.5 |

## Files and reproduction

- `output/kodak/kodimXX/` contains `gray.png` (the original) and, for each method, `<method>_recon.png` and `<method>.log` (all printed output: spike counts, MSE, PSNR, active).
- `output/kodak/results.csv` has every number above, one row per image and method.

To regenerate a single result:

```
python sparse_code_image.py --method <mean|spiking_lca|latency_slca|temporal_pattern_slca> --preprocess whiten \
    --image images/kodak/kodimXX.png --out-dir output/kodak > output/kodak/kodimXX/<method>.log
```

## Rounding error

In practice, the latency and temporal coding methods' discretization of values needs correction by removing the hard reset of v at the beginning of each cycle, which results in correction over time. For example, in latency coding, if the true u_i - lambda * lr is 4.5dt, the true firing timestep is timestep S - 4.5, but it will only spike at timestep S - 4.0, as it only checks whether the threshold is breached at integer time steps. Thus, it fires 0.5 timesteps too late and represents a value of 4.0dt, 0.5dt too low. A similar rounding error applies to temporal coding LCA, where a value such as 8.2q would be represented as a spike train of [...,1, 0, 0, 0] = 8q, meaning the represented is again slightly too low. Thus, we correct for this rounding by carrying over the leftover v membrane potential to the next cycle. Over the next few cycles, this extra membrane potential contributes to a correction to the rounding. With u_i - lr*lamdba = 4.5dt, a carrying the leftover v over instead of overwriting would result in v having a membrane potential of 0.5dt after the cycle finishes, carrying extra potential to the next cycle. After the next few cycles, the 0.5dt extra potential will cause some spikes to have slightly earlier firing times, correcting for the slightly late firing time of the original rounding error by adding that missing value to the state u_i in the next cycles. However, since u_i is a leaky integrator, this correction is not exact, as recent cycle leftovers are weighted higher than older ones. 

See `errorboundcalculationnotes.png` for informal calculations; formalization will be added soon!
