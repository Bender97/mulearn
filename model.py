# Copyright (c) Daniel Fusaro. All rights reserved.
# -------------------------------
# Model Definition
# -------------------------------

from utils.math_functions import leaky_relu, leaky_relu_deriv, softmax, argmax
from linear import Linear
import math


def normalize(x, eps=1e-8):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


class SimpleNN:
    def __init__(self, input_size, num_classes, queue_size = 10):
        self.fc1 = Linear(input_size, 8)
        self.fc2 = Linear(8, 16)
        self.fc3 = Linear(16, num_classes)
        self.queue = [-i for i in range(1, queue_size + 1)]
        self.cont = 0

    def forward(self, x):
        x = [d/1023.0 for d in x]  # Normalize input data

        x1 = self.fc1.forward(x)
        x1 = normalize(x1)                 # normalize before activation
        self.x1 = [leaky_relu(v) for v in x1]

        x2 = self.fc2.forward(self.x1)
        x2 = normalize(x2)                 # normalize before activation
        self.x2 = [leaky_relu(v) for v in x2]

        logits = self.fc3.forward(self.x2)
        probs = softmax(logits)
        self.probs = probs
        return probs

    def predict(self, x):

        probs = self.forward(x)
        color_idx = argmax(probs)
        predicted_confidence = probs[color_idx]

        self.queue[self.cont] = color_idx
        self.cont = (self.cont + 1) % len(self.queue)

        for i in range(1, len(self.queue)):
            if self.queue[i] != self.queue[0]:
                if self.queue[i] >= 0:
                    self.queue = [-i for i in range(1, len(self.queue) + 1)]
                    self.cont = 0
                return -1, 0.0
            
        return color_idx, predicted_confidence

    def backward(self, target_index, lr):
        grad_out = [p for p in self.probs]
        grad_out[target_index] -= 1.0# dL/dz = p - y (y is 1 at target_index)
        grad_fc3 = self.fc3.backward(grad_out, lr)
        grad_x2 = [grad_fc3[i] * leaky_relu_deriv(self.x2[i]) for i in range(len(self.x2))]
        grad_fc2 = self.fc2.backward(grad_x2, lr)
        grad_x1 = [grad_fc2[i] * leaky_relu_deriv(self.x1[i]) for i in range(len(self.x1))]
        _ = self.fc1.backward(grad_x1, lr)