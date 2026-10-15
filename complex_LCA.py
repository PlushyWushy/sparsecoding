"""
Complex LCA with slowness penalty
TODO:
Test on actual movies
"""

import numpy as np
lr = 0.0027
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128)) + rng.standard_normal((64, 128))*1j
beta = 0.1
image = rng.standard_normal(64) + rng.standard_normal(64) * 1j
epsilon = 0.1
def LCA(input, lamby, Phi, beta, lr, epsilon, num_bases):
    x = input
    I = np.eye(num_bases)
    a = np.zeros(num_bases)
    recurrent = (I - lr * ((Phi.transpose().conj() @ Phi)) - lr * 2 * beta * I) * epsilon
    feedforward = lr * Phi.transpose().conj() @ x * epsilon
    u = np.zeros(128)
    for i in range(10000):
        u = recurrent @ a + feedforward + (1 - epsilon) * u + epsilon * lr * 2 * beta * np.abs(a) * np.sign(a) #still image, so c_t-1 = c_t
        a = np.sign(u) * np.maximum(np.abs(u) - lr*lamby, 0)
    return a
if __name__ == "__main__":
    print(LCA(image, lamby, Phi, beta, lr, epsilon, 128))