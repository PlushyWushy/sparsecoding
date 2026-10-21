# Sparse Coding for Spiking Neural networks

See [`SNN/latency_SLCA.py`](SNN/latency_SLCA.py) for latest implementation, and see [`AlternativeCodingMethods.pdf`](AlternativeCodingMethods.pdf) for the formulation.

---

## Results from latency vs rate coded LCAs with lambda = 4:

### latency:

200000 steps (dt=0.0174, 3489 time units)

5057 recurrent spikes/patch, 24296 input spikes/patch, 409.8s

- **MSE:** 0.000455
- **PSNR:** 33.42 dB
- **active coefficients per patch:** 12.8 / 128

### temporal pattern:

3600 steps (q=0.017, 2^9 = 512, so roughly same representational power as latency and rate), 8441 recurrent spikes/patch, 55481 input spikes/patch, 5.1s

- **MSE:** 0.000451
- **PSNR:** 33.46 dB
- **active coefficients per patch:** 12.9 / 128

### rate:

200000 steps (dt=0.115, 22929 time units)

40945 recurrent spikes/patch, 646003 input spikes/patch, 407.1s

- **MSE:** 0.000251
- **PSNR:** 36.01 dB
- **active coefficients per patch:** 16.1 / 128

---

## Results from latency vs rate coded LCAs with lambda = 1:

### latency:

200000 steps (dt=0.0174, 3489 time units)

8517 recurrent spikes/patch, 24296 input spikes/patch, 413.9s

- **MSE:** 0.000234
- **PSNR:** 36.31 dB
- **active coefficients per patch:** 21.8 / 128

### temporal pattern:

3600 steps (q=0.017, 2^9 = 512, so roughly same representational power as latency and rate), 13515 recurrent spikes/patch, 55481 input spikes/patch, 5.0s

- **MSE:** 0.000230
- **PSNR:** 36.39 dB
- **active coefficients per patch:** 22.0 / 128

### rate:

200000 steps (dt=0.115, 22929 time units)

65695 recurrent spikes/patch, 646003 input spikes/patch, 413.3s

- **MSE:** 0.000043
- **PSNR:** 43.65 dB
- **active coefficients per patch:** 38.8 / 128

---

## To match number of coefficients, rate coding was run with lambda = 2.5:

### rate:

200000 steps (dt=0.115, 22929 time units), 50266 recurrent spikes/patch, 646003 input spikes/patch, 264.4s

- **MSE:** 0.000139
- **PSNR:** 38.57 dB
- **active coefficients per patch:** 22.0 / 128
