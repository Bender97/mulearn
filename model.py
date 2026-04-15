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
    def __init__(self, neurons, queue_size = 10):
        # neuron is a list, where each element is the number of neurons in that layer
        # always it is [input_size, h_1, h_2, ..., h_n, num_classes]

        self.fcs = []
        for i in range(len(neurons) - 1):
            self.fcs.append(Linear(neurons[i], neurons[i + 1]))
        
        self.queue = [-i for i in range(1, queue_size + 1)]
        self.cont = 0
        self.outputs = []

    def forward(self, x):
        self.outputs = []
        
        x = [d/1023.0 for d in x]  # Normalize input data
        
        last_output = x
        for i in range(len(self.fcs) - 1):
            x = self.fcs[i].forward(last_output)
            x = normalize(x)
            last_output = [leaky_relu(v) for v in x]
            self.outputs.append(last_output)

        logits = self.fcs[-1].forward(last_output)
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
        grad_out[target_index] -= 1.0 # dL/dz = p - y (y is 1 at target_index)


        grad = grad_out
        for i in reversed(range(len(self.fcs) - 1)):
            grad = self.fcs[i + 1].backward(grad, lr)
            grad = [grad[j] * leaky_relu_deriv(self.outputs[i][j]) for j in range(len(self.outputs[i]))]
        
        _ = self.fcs[0].backward(grad, lr)
        

if __name__ == "__main__":
    INPUT_SIZE  = 3
    NUM_CLASSES = 4
    model = SimpleNN([INPUT_SIZE, 8, 16, NUM_CLASSES])
    input_data = [512, 256, 128]
    target_index = 2
    lr = 0.01
    
    # overfit a single example
    for epoch in range(100):
        probs = model.forward(input_data)
        print("Epoch {:d} - probs {}".format(epoch, probs))
        model.backward(target_index, lr)