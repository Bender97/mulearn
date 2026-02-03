# Copyright (c) Daniel Fusaro. All rights reserved.

import math
from utils.general_functions import MiniRand

class Linear:
    def __init__(self, in_f, out_f):
        self.in_f, self.out_f = in_f, out_f
        # Xavier (Glorot) uniform initialization
        r = MiniRand(seed=0)
        limit = math.sqrt(6.0 / (in_f + out_f))
        self.W = [[r.uniform(-limit, limit) for _ in range(in_f)] for _ in range(out_f)]
        self.b = [0.0 for _ in range(out_f)]  # usually biases start at zero

    def forward(self, x):
        self.x = x
        y = []
        for i in range(self.out_f):
            s = self.b[i]
            for j in range(self.in_f):
                s += self.W[i][j] * x[j]
            y.append(s)
        return y

    def backward(self, grad_out, lr):
        grad_in = [0.0 for _ in range(self.in_f)]
        for i in range(self.out_f):
            for j in range(self.in_f):
                grad = grad_out[i] * self.x[j]
                self.W[i][j] -= lr * grad
                grad_in[j] += grad_out[i] * self.W[i][j]
            self.b[i] -= lr * grad_out[i]
        return grad_in
