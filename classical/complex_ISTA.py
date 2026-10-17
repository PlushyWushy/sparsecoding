"""
Complex ISTA with slowness penalty
TODO:
Test on actual movies
"""
import numpy as np


lr = 0.001
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128)) + rng.standard_normal((64, 128))*1j
beta = 0.1
image = rng.standard_normal(64) + rng.standard_normal(64) * 1j
def ISTA(input, lamby, Phi, beta, lr, num_bases):
    x = input
    I = np.eye(num_bases)
    a = np.zeros(num_bases)
    recurrent = (I - lr * ((Phi.conj().transpose() @ Phi)) - lr*2*beta * I)
    feedforward = lr * Phi.conj().transpose() @ x
    for i in range(1000):
        z = recurrent @ a + feedforward + lr * 2 * beta * np.abs(a) * np.sign(a) #still image, so c_t-1 = c_t
        a = np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0) #np.sign(z) returns z/|z| = a/|a|
    return a
if __name__ == "__main__":
    print(ISTA(image, lamby, Phi, beta, lr, 128))