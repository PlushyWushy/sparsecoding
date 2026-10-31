import numpy as np

lr = 0.001
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
image = rng.standard_normal(64)
def ISTA(input, lamby, Phi, lr, num_bases, steps=1000):
    x = input
    I = np.eye(num_bases)
    recurrent = (I - lr * ((Phi.transpose() @ Phi)))
    feedforward = lr * Phi.transpose() @ x
    a = np.zeros_like(feedforward) #same shape as feedforward, so x can be one patch or a matrix of patches (one per column)
    for i in range(steps):
        z = recurrent @ a + feedforward
        a = np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0)
    return a
if __name__ == "__main__":
    print(ISTA(image, lamby, Phi, lr, 128))
