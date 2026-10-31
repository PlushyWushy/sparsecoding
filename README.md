# Sparse Coding for Spiking Neural networks

See [`AlternativeCodingMethods.pdf`](AlternativeCodingMethods.pdf) for formulations.

---

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
| Latency | 400 cycles × 512 steps = 204,800 steps, T = 99.9th percentile of \|x\|, leftover v carried across cycles (see Rounding error) |
| Temporal pattern | 400 cycles × 9 bits = 3,600 steps, same T, so q = T/512, the same grid as latency, leftover v carried across cycles |

Coded images are decoded from the final state. "Active" is the number of nonzero coefficients per patch in the doubled (±Φ) dictionary.

## Summary (mean ± std over 24 images)

| Method | PSNR (dB) | Gain over control (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch | Steps | Wall-clock / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control (mean only) | 22.67 ± 2.44 | — | 0 | — | — | — | — |
| Rate | **43.40 ± 2.33** | **20.73 ± 0.92** | 38.7 ± 8.0 | 65,068 | 1,338,629 | 204,800 | 731 s |
| Latency | 42.50 ± 2.34 | 19.83 ± 0.97 | 40.1 ± 10.0 | **17,941** | **24,952** | 204,800 | 929 s |
| Temporal pattern | 42.49 ± 2.32 | 19.83 ± 0.96 | 40.0 ± 10.0 | 25,546 | 72,266 | **3,600** | **10.9 s** |

Wall-clock was measured with 6 jobs running at once, 3 threads each (latency for `kodim01`–`kodim12` alongside other jobs). Results for rate code at λ = 2.5, and for latency / temporal pattern before the leftover carry, are in `legacy_tests/rate_lambda2.5/`.

## Per-image PSNR (dB) and active coefficients per patch

| Image | Control | Rate | Latency | TP | Active (rate) | Active (lat / TP) |
|---|---:|---:|---:|---:|---:|---:|
| kodim01 | 20.64 | 41.02 | 40.13 | 40.13 | 48.7 | 53.8 / 53.7 |
| kodim02 | 27.55 | 45.50 | 44.61 | 44.59 | 43.8 | 40.0 / 40.0 |
| kodim03 | 25.65 | 46.77 | 45.89 | 45.86 | 30.9 | 29.8 / 29.7 |
| kodim04 | 25.02 | 44.35 | 43.65 | 43.60 | 40.8 | 41.5 / 41.1 |
| kodim05 | 18.49 | 39.31 | 38.53 | 38.44 | 46.2 | 51.1 / 50.7 |
| kodim06 | 22.01 | 42.78 | 41.71 | 41.79 | 42.4 | 44.5 / 44.8 |
| kodim07 | 22.44 | 44.52 | 43.84 | 43.88 | 30.6 | 33.8 / 33.9 |
| kodim08 | 16.92 | 38.20 | 37.51 | 37.56 | 40.3 | 44.8 / 44.9 |
| kodim09 | 23.44 | 45.07 | 44.38 | 44.25 | 28.8 | 27.5 / 27.0 |
| kodim10 | 24.14 | 45.07 | 44.15 | 44.20 | 33.3 | 31.9 / 32.1 |
| kodim11 | 22.66 | 43.47 | 42.41 | 42.50 | 39.7 | 40.6 / 40.9 |
| kodim12 | 23.97 | 45.07 | 44.21 | 44.12 | 32.7 | 30.5 / 30.2 |
| kodim13 | 19.56 | 39.22 | 38.14 | 38.17 | 55.2 | 62.2 / 62.2 |
| kodim14 | 21.55 | 41.92 | 41.04 | 41.06 | 48.3 | 52.4 / 52.4 |
| kodim15 | 21.88 | 43.70 | 42.71 | 42.83 | 28.0 | 26.2 / 26.5 |
| kodim16 | 26.10 | 46.66 | 45.78 | 45.68 | 43.2 | 46.2 / 45.7 |
| kodim17 | 23.49 | 43.89 | 43.11 | 43.10 | 37.2 | 38.7 / 38.7 |
| kodim18 | 21.83 | 41.89 | 40.70 | 40.84 | 47.1 | 49.1 / 49.5 |
| kodim19 | 21.38 | 42.57 | 41.72 | 41.78 | 36.0 | 36.8 / 37.0 |
| kodim20 | 22.44 | 44.76 | 43.86 | 43.73 | 23.5 | 23.7 / 23.3 |
| kodim21 | 21.77 | 42.66 | 41.79 | 41.75 | 35.5 | 37.3 / 37.0 |
| kodim22 | 24.41 | 44.42 | 43.26 | 43.32 | 46.1 | 46.7 / 46.8 |
| kodim23 | 25.39 | 46.60 | 45.68 | 45.66 | 29.0 | 27.0 / 27.0 |
| kodim24 | 21.31 | 42.14 | 41.12 | 41.03 | 42.2 | 45.9 / 45.4 |

## Files and reproduction

- `output/kodak/kodimXX/` contains `gray.png` (the original) and, for each method, `<method>_recon.png` and `<method>.log` (all printed output: spike counts, MSE, PSNR, active).
- `output/kodak/results.csv` has every number above, one row per image and method.

To regenerate a single result:

```
python sparse_code_image.py --method <mean|spiking_lca|latency_slca|temporal_pattern_slca> \
    --image images/kodak/kodimXX.png --out-dir output/kodak > output/kodak/kodimXX/<method>.log
```

## Rounding error

In practice, the latency and temporal coding methods' discretization of values needs correction by removing the hard reset of v at the beginning of each cycle, which results in correction over time. For example, in latency coding, if the true u_i - lambda * lr is 4.5dt, the true firing timestep is timestep S - 4.5dt, but it will only spike at timestep S - 4.0dt, as it only checks whether the threshold is breached at integer time steps. Thus, it fires 0.5 timesteps too late and represents a value of 4.0dt, 0.5dt too low. A similar rounding error applies to temporal coding LCA, where a value such as 8.2q would be represented as a spike train of [...,1, 0, 0] = 8q, meaning the represented is again slightly too low. Thus, we correct for this rounding by carrying over the leftover v membrane potential to the next cycle. Over the next few cycles, this extra membrane potential contributes to a correction to the rounding. With u_i - lr*lamdba = 4.5dt, a carrying the leftover v over instead of overwriting would result in v having a membrane potential of 0.5dt after the cycle finishes, carrying extra potential to the next cycle. After the next few cycles, the 0.5dt extra potential will cause some spikes to have slightly earlier firing times, correcting for the slightly late firing time of the original rounding error by adding that missing value to the state u_i in the next cycles. However, since u_i is a leaky integrator, this correction is not exact, as recent cycle leftovers are weighted higher than older ones. 