# Copyright (c) Daniel Fusaro. All rights reserved.
# -------------------------
# Loss function
# -------------------------

import math

def cross_entropy_loss(pred, target_index):
    # pred: probabilities
    eps = 1e-9
    loss = -math.log(pred[target_index] + eps)
    if math.isnan(loss):
        print("aaaa ho un errore troppo grande -> prova un LR più piccolo")
        quit(1)
    return loss