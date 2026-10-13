import numpy as np
import torch.nn as nn
I = np.eye(128)
lr = 0.001
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
print(Phi)
image = rng.standard_normal(64)
epsilon = 0.1
class SimpleNeuralNet(nn.Module):
    def __init__(self, Phi, lr, lamby):
        super(SimpleNeuralNet, self).__init__()
        self.recurrent = lambda a: (I - lr * ((Phi.transpose() @ Phi))) @ a
        self.feedforward = lambda x: lr * Phi.transpose() @ x
        self.activation = lambda z: np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0)                       # Activation function
        
    def forward(self, x, epsilon):
        a = np.zeros(128)
        a_next = np.zeros(128)
        u = np.zeros(128)
        for i in range(10000):
            a = a_next
            u = epsilon * (self.recurrent(a) + self.feedforward(x)) + (1 - epsilon) * u
            a_next = self.activation(u)
            if np.linalg.norm(a_next - a) < 1e-12:
                break
        return a_next
model = SimpleNeuralNet(Phi, lr, lamby)
print(model(image, epsilon))