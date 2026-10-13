import numpy as np

I = np.eye(128)
lr = 0.001
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
print(Phi)
image = rng.standard_normal(64)
def ISTA(input):
    x = input
    a = np.zeros(128)
    recurrent = (I - lr * ((Phi.transpose() @ Phi)))
    feedforward = lr * Phi.transpose() @ x
    for i in range(1000):
        z = recurrent @ a + feedforward
        a = np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0)
    return a
print(ISTA(image))