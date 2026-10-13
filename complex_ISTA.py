"""
Complex ISTA with slowness penalty
TODO:
Test on actual movies
"""
import numpy as np

I = np.eye(128)
lr = 0.001
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128)) + rng.standard_normal((64, 128))*1j
beta = 0.1
image = rng.standard_normal(64) + rng.standard_normal(64) * 1j
def ISTA(input, lamby, Phi, beta, I, lr):
    x = input
    a = np.zeros(128)
    recurrent = (I - lr * ((Phi.conj().transpose() @ Phi)) - lr*2*beta * I)
    feedforward = lr * Phi.conj().transpose() @ x
    for i in range(1000):
        z = recurrent @ a + feedforward + lr * 2 * beta * np.abs(a) * np.sign(a) #still image, so c_t-1 = c_t
        a = np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0) #np.sign(z) returns z/|z| = a/|a|
    return a
if __name__ == "__main__":
    print(ISTA(input, lamby, Phi, beta, I, lr))