Sparse Coding for Spiking Neural networks

See SNN/latency_SLCA.py for latest implementation, and see Temporal_Coded_Spiking_LCA.pdf for the formulation.

Results from latency vs rate coded LCAs with lambda = 4:


latency: 200000 steps (dt=0.0174, 3489 time units), 5057 recurrent spikes/patch, 24296 input spikes/patch, 409.8s
MSE: 0.000455   
PSNR: 33.42 dB
active coefficients per patch: 12.8 / 128


rate: 200000 steps (dt=0.115, 22929 time units), 40945 recurrent spikes/patch, 646003 input spikes/patch, 407.1s
MSE: 0.000251  
PSNR: 36.01 dB
active coefficients per patch: 16.1 / 128