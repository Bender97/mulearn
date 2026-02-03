# Copyright (c) Daniel Fusaro. All rights reserved.
# -------------------------------
# Utility functions
# -------------------------------

import math

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_deriv(x, alpha=0.01):
    return 1 if x > 0 else alpha

def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def argmax(lst):
    idx, val = 0, lst[0]
    for i, v in enumerate(lst):
        if v > val:
            idx, val = i, v
    return idx