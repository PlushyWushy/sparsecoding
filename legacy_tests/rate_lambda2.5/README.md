# Rate code at λ = 2.5 (legacy)

Rate-coded spiking LCA (`spiking_lca`) at λ = 2.5 on all 24 Kodak images, run to compare against the latency and temporal-pattern codes as they were before the rounding-remainder carry was added ("pre-carry": v overwritten at the start of every cycle, round-to-nearest via `+ dt/2` / `+ q/2`). Setup is otherwise identical to the main comparison in `README.md`: same images, preprocessing, Φ, ε = 0.1, lr = 0.00148, 204,800 steps.

## Summary (mean ± std over 24 images)

| Method | PSNR (dB) | Gain over control (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch |
|---|---:|---:|---:|---:|---:|
| Rate, λ = 1 | 43.40 ± 2.33 | 20.73 ± 0.92 | 38.7 ± 8.0 | 65,068 | 1,338,629 |
| Rate, λ = 2.5 | 38.26 ± 2.83 | 15.60 ± 1.12 | 23.8 ± 5.9 | 51,787 | 1,338,629 |
| Latency, pre-carry (λ = 1) | 35.45 ± 2.40 | 12.78 ± 1.23 | 31.0 ± 10.2 | 12,215 | 24,952 |
| Temporal pattern, pre-carry (λ = 1) | 35.45 ± 2.40 | 12.78 ± 1.22 | 31.0 ± 10.3 | 18,098 | 72,266 |

Paired PSNR differences (same image, 95% CI):

| Comparison | Mean difference (dB) | Images where the first is higher |
|---|---:|---:|
| Rate, λ = 2.5 − Latency, pre-carry (λ = 1) | +2.81 ± 0.27 | 24 / 24 |
| Rate, λ = 2.5 − Temporal pattern, pre-carry (λ = 1) | +2.81 ± 0.27 | 24 / 24 |
| Rate, λ = 2.5 − Rate, λ = 1 | -5.13 ± 0.25 | 0 / 24 |

Wall-clock for the λ = 2.5 runs: 872 s / image on average (4 jobs at once, 3 threads each, alongside other jobs).

## Per-image PSNR (dB) and active coefficients per patch

| Image | Rate λ = 1 | Rate λ = 2.5 | Latency pre-carry | TP pre-carry | Active (λ = 1) | Active (λ = 2.5) | Active (lat / TP pre-carry) |
|---|---:|---:|---:|---:|---:|---:|---:|
| kodim01 | 41.02 | 34.99 | 33.06 | 33.06 | 48.7 | 31.2 | 46.0 / 46.0 |
| kodim02 | 45.50 | 41.47 | 37.68 | 37.69 | 43.8 | 25.5 | 20.0 / 20.0 |
| kodim03 | 46.77 | 42.17 | 38.89 | 38.89 | 30.9 | 18.1 | 21.2 / 21.2 |
| kodim04 | 44.35 | 39.98 | 36.92 | 36.91 | 40.8 | 24.3 | 28.9 / 28.8 |
| kodim05 | 39.31 | 33.44 | 31.32 | 31.30 | 46.2 | 29.4 | 45.3 / 45.2 |
| kodim06 | 42.78 | 37.03 | 34.07 | 34.10 | 42.4 | 27.0 | 32.9 / 33.0 |
| kodim07 | 44.52 | 39.60 | 37.81 | 37.83 | 30.6 | 19.3 | 34.4 / 34.5 |
| kodim08 | 38.20 | 32.60 | 31.61 | 31.62 | 40.3 | 24.2 | 44.5 / 44.6 |
| kodim09 | 45.07 | 40.59 | 37.55 | 37.52 | 28.8 | 16.3 | 20.6 / 20.5 |
| kodim10 | 45.07 | 40.56 | 37.42 | 37.44 | 33.3 | 19.2 | 24.6 / 24.7 |
| kodim11 | 43.47 | 38.01 | 35.11 | 35.14 | 39.7 | 24.2 | 29.7 / 29.8 |
| kodim12 | 45.07 | 40.32 | 37.10 | 37.09 | 32.7 | 17.8 | 17.7 / 17.7 |
| kodim13 | 39.22 | 33.09 | 30.87 | 30.87 | 55.2 | 37.0 | 51.9 / 52.0 |
| kodim14 | 41.92 | 36.21 | 33.84 | 33.85 | 48.3 | 30.2 | 42.7 / 42.7 |
| kodim15 | 43.70 | 38.91 | 35.73 | 35.75 | 28.0 | 15.5 | 17.0 / 17.0 |
| kodim16 | 46.66 | 41.07 | 38.16 | 38.13 | 43.2 | 27.6 | 35.3 / 35.2 |
| kodim17 | 43.89 | 39.33 | 36.80 | 36.80 | 37.2 | 22.3 | 32.5 / 32.6 |
| kodim18 | 41.89 | 36.33 | 33.00 | 33.03 | 47.1 | 30.6 | 37.2 / 37.3 |
| kodim19 | 42.57 | 37.30 | 34.88 | 34.89 | 36.0 | 21.2 | 27.0 / 27.1 |
| kodim20 | 44.76 | 39.96 | 36.41 | 36.39 | 23.5 | 14.6 | 16.5 / 16.5 |
| kodim21 | 42.66 | 37.48 | 34.57 | 34.56 | 35.5 | 23.0 | 30.2 / 30.1 |
| kodim22 | 44.42 | 39.00 | 35.36 | 35.37 | 46.1 | 29.1 | 31.4 / 31.4 |
| kodim23 | 46.60 | 42.56 | 39.07 | 39.06 | 29.0 | 16.2 | 19.4 / 19.4 |
| kodim24 | 42.14 | 36.33 | 33.56 | 33.54 | 42.2 | 27.7 | 37.8 / 37.7 |

## Files and reproduction

- `legacy_tests/rate_lambda2.5/kodimXX/`: `spiking_lca_recon.png` and `spiking_lca.log` for the λ = 2.5 run (originals: `output/kodak/kodimXX/gray.png`).
- `legacy_tests/rate_lambda2.5/results.csv`: every number above, including the control, rate λ = 1 and the pre-carry latency/TP rows.

λ was overridden after import, because editing `lamby` in `SNN/config.py` changes the Φ cache key and would relearn Φ:

```
python -c "import sys, runpy; import SNN.config as c; c.lamby = 2.5; \
sys.argv = ['x', '--method', 'spiking_lca', '--image', 'images/kodak/kodimXX.png', '--out-dir', '<tmp dir>']; \
runpy.run_path('sparse_code_image.py', run_name='__main__')"
```
