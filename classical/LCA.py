import numpy as np

lr = 0.0027
lamby = 1.0
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
image = rng.standard_normal(64)
epsilon = 0.1
def LCA(input, lamby, Phi, lr, epsilon, num_bases):
    x = input
    I = np.eye(num_bases)
    recurrent = (I - lr * ((Phi.transpose() @ Phi))) * epsilon
    feedforward = lr * Phi.transpose() @ x * epsilon
    a = np.zeros_like(feedforward) #same shape as feedforward, so x can be one patch or a matrix of patches (one per column)
    u = np.zeros_like(feedforward)
    for i in range(10000):
        u = recurrent @ a + feedforward + (1 - epsilon) * u
        a = np.sign(u) * np.maximum(np.abs(u) - lr*lamby, 0)
    return a
if __name__ == "__main__":
    print(LCA(image, lamby, Phi, lr, epsilon, 128))
