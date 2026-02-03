# Copyright (c) Daniel Fusaro. All rights reserved.
# -------------------------------
# Training / Evaluation
# -------------------------------

from utils.math_functions import argmax
from utils.loss_function import cross_entropy_loss

def accuracy(model, data, labels):
    correct = 0
    for i in range(len(data)):
        probs = model.forward(data[i])
        pred_class = argmax(probs)
        if pred_class == labels[i]:
            correct += 1
    return correct / len(data)

def accuracy_and_loss(model, data, labels):
    correct = 0
    total_loss = 0
    for i in range(len(data)):
        probs = model.forward(data[i])
        total_loss += cross_entropy_loss(probs, labels[i])
        pred_class = argmax(probs)
        if pred_class == labels[i]:
            correct += 1
    return correct / len(data) * 100, total_loss / len(data)

def get_confusion_matrix(model, data, labels, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(len(data)):
        probs = model.forward(data[i])
        pred_class = argmax(probs)
        true_class = labels[i]
        matrix[true_class][pred_class] += 1
    return matrix