# Copyright (c) Daniel Fusaro. All rights reserved.
# -------------------------
# Training utilities
# -------------------------

def shuffled_copy(lst, seed=2):
    random = MiniRand(seed=seed)

    lst_copy = lst[:]# make a shallow copy
    for i in range(len(lst_copy) - 1, 0, -1):
        j = int(random.rand() * (i + 1))
        lst_copy[i], lst_copy[j] = lst_copy[j], lst_copy[i]
    return lst_copy

def describe(y, msg, NUM_CLASSES):
    print("{:10s} set: (containing {:d} elements)".format(msg, len(y)))
    print("Class", ("{:^7d}"*NUM_CLASSES).format(*[i for i in range(NUM_CLASSES)]))
    counts = [0 for _ in range(NUM_CLASSES)]
    for label in y:
        counts[label] += 1
    print("Train", ("{:^7d}"*NUM_CLASSES).format(*[y.count(i) for i in range(NUM_CLASSES)]))

def print_confusion_matrix(cm):
    print("Confusion matrix:")
    for row in cm:
        print(row)

# portable_random.py
class MiniRand:
    def __init__(self, seed=123):
        self.state = seed & 0xFFFFFFFF

    def rand(self):
        # same sequence on both CPython and MicroPython
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0

    def uniform(self, a, b):
        return a + (b - a) * self.rand()
