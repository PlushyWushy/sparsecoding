import numpy as np
import torch.nn as nn
I = np.eye(128)
lr = 0.001
lamby = 0.4
rng = np.random.default_rng(0)
Phi = rng.standard_normal((64, 128))
print(Phi)
image = rng.standard_normal(64)
class SimpleNeuralNet(nn.Module):
    def __init__(self, Phi, lr, lamby):
        super(SimpleNeuralNet, self).__init__()
        self.recurrent = lambda a: (I - lr * ((Phi.transpose() @ Phi))) @ a
        self.feedforward = lambda x: lr * Phi.transpose() @ x
        self.activation = lambda z: np.sign(z) * np.maximum(np.abs(z) - lr*lamby, 0)                       # Activation function

        
    def forward(self, x):
        a = np.zeros(128)
        a_next = self.activation(self.recurrent(a) + self.feedforward(x))
        for i in range(1000):
            a = a_next
            a_next = self.activation(self.recurrent(a) + self.feedforward(x))
            if np.linalg.norm(a_next - a) < 1e-12:
                break
        return a
model = SimpleNeuralNet(Phi, lr, lamby)
print(model(image))