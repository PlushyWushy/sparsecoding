import numpy as np

I = np.eye(128)
lr = 0.0027
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
print(Phi)
image = rng.standard_normal(64)
epsilon = 0.1
def LCA(input):
    x = input
    a = np.zeros(128)
    recurrent = (I - lr * ((Phi.transpose() @ Phi))) * epsilon
    feedforward = lr * Phi.transpose() @ x * epsilon
    u = np.zeros(128)
    for i in range(10000):
        u = recurrent @ a + feedforward + (1 - epsilon) * u
        a = np.sign(u) * np.maximum(np.abs(u) - lr*lamby, 0)
    return a
print(LCA(image))