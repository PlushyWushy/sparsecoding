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
| Dictionary | `SNN/phi_learned_whiten.npz`: 64×128, learned on 100k random patches of whitened Kodak images, columns at norm 1 |
| Parameters | λ = 0.1, ε = 0.1, lr = 1/‖Φ‖₂² = 0.160 (`SNN/config.py`) |
| Rate | 204,800 steps, dt = 1 / max\|x\| per image |
| Latency | 400 cycles × 512 steps = 204,800 steps, T = max\|x\| per image, leftover v carried across cycles (see Rounding error) |
| Temporal pattern | 400 cycles × 9 bits = 3,600 steps, same T, so q = T/512, the same grid as latency, leftover v carried across cycles |

PSNR = 10·log10(1/MSE) on pixels in [0, 1]. "Active" is the number of nonzero coefficients per patch in the doubled (±Φ) dictionary.

## Summary (mean ± std over 24 images)

| Method | PSNR (dB) | Difference from control (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch | Steps | Wall-clock / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control (mean only) | 15.12 ± 2.47 | — | 0 | — | — | — | — |
| Rate | 13.59 ± 2.84 | −1.53 ± 3.02 | 60.1 ± 8.0 | 372,690 | 860,211 | 204,800 | 783 s |
| Latency | 14.38 ± 2.87 | −0.74 ± 3.12 | 48.8 ± 7.9 | 23,799 | 25,011 | 204,800 | 862 s |
| Temporal pattern | 14.38 ± 2.88 | −0.73 ± 3.12 | 48.7 ± 7.8 | 52,277 | 65,999 | 3,600 | 13.1 s |

Wall-clock was measured with 6 jobs running at once, 3 threads each. Earlier results are in `legacy_tests/`. 

## Per-image PSNR (dB) and active coefficients per patch

| Image | Control | Rate | Latency | TP | Active (rate) | Active (lat / TP) |
|---|---:|---:|---:|---:|---:|---:|
| kodim01 | 16.52 | 10.50 | 11.48 | 11.48 | 66.4 | 58.5 / 58.5 |
| kodim02 | 22.24 | 16.72 | 17.28 | 17.27 | 67.4 | 51.3 / 51.2 |
| kodim03 | 16.31 | 17.54 | 18.31 | 18.34 | 54.1 | 42.2 / 42.1 |
| kodim04 | 16.67 | 16.88 | 17.52 | 17.57 | 65.3 | 51.3 / 51.1 |
| kodim05 | 14.67 | 8.99 | 9.85 | 9.83 | 68.1 | 57.4 / 57.3 |
| kodim06 | 12.70 | 12.51 | 13.09 | 13.10 | 63.4 | 52.2 / 52.2 |
| kodim07 | 16.89 | 14.32 | 15.64 | 15.63 | 52.1 | 41.5 / 41.5 |
| kodim08 | 12.44 | 8.04 | 9.00 | 8.98 | 59.4 | 52.4 / 52.4 |
| kodim09 | 16.68 | 16.22 | 16.53 | 16.58 | 47.8 | 37.4 / 37.5 |
| kodim10 | 16.43 | 15.99 | 16.82 | 16.86 | 56.5 | 42.8 / 42.8 |
| kodim11 | 15.92 | 13.03 | 13.72 | 13.69 | 61.8 | 50.0 / 50.0 |
| kodim12 | 15.04 | 16.47 | 17.84 | 17.78 | 56.0 | 44.4 / 44.4 |
| kodim13 | 14.09 | 8.75 | 8.93 | 8.92 | 73.8 | 65.1 / 65.0 |
| kodim14 | 14.72 | 11.92 | 12.87 | 12.87 | 71.2 | 59.5 / 59.4 |
| kodim15 | 9.85 | 13.87 | 15.69 | 15.70 | 52.4 | 41.2 / 41.1 |
| kodim16 | 15.63 | 16.50 | 17.09 | 17.11 | 64.6 | 53.3 / 53.2 |
| kodim17 | 14.47 | 15.26 | 16.10 | 16.12 | 61.9 | 47.8 / 47.8 |
| kodim18 | 16.70 | 11.39 | 12.25 | 12.29 | 68.7 | 56.3 / 56.2 |
| kodim19 | 14.90 | 12.80 | 13.33 | 13.34 | 54.6 | 44.1 / 44.0 |
| kodim20 | 9.38 | 13.48 | 14.15 | 14.10 | 42.4 | 33.4 / 33.4 |
| kodim21 | 15.86 | 12.67 | 13.46 | 13.41 | 54.5 | 45.3 / 45.3 |
| kodim22 | 16.09 | 14.15 | 15.01 | 15.03 | 69.0 | 55.3 / 55.2 |
| kodim23 | 14.84 | 17.84 | 18.33 | 18.36 | 48.1 | 36.3 / 36.3 |
| kodim24 | 13.74 | 10.34 | 10.80 | 10.79 | 62.0 | 51.9 / 51.8 |

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
