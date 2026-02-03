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

def rgb2hsv(rgb):
    
    rn, gn, bn = rgb
    
    #normalize
    rn /= 1023
    gn /= 1023
    bn /= 1023
    
    mx = rn if rn >= gn else gn
    mx = mx if mx >= bn else bn
    mn = rn if rn <= gn else gn
    mn = mn if mn <= bn else bn

    # Value
    v = mx
    delta = mx - mn

    # Hue
    if delta == 0:
        h = 0.0
    else:
        if mx == rn:
            # (gn - bn) / delta mod 6
            h = ( (gn - bn) / delta ) % 6.0
        elif mx == gn:
            h = ( (bn - rn) / delta ) + 2.0
        else:# mx == bn
            h = ( (rn - gn) / delta ) + 4.0

        h = 60.0 * h

        # Ensure hue in [0,360)
        while h < 0:
            h += 360.0
        # If exactly 360 make it 0
        while h >= 360.0:
            h -= 360.0

    # Saturation
    if mx == 0:
        s = 0.0
    else:
        s = delta / mx

    h/= 360.0# normalize hue to [0,1]

    return [h*1023, s*1023, v*1023]

def data_rgb2hsv(data):
    for i in range(len(data)):
        data[i] = rgb2hsv(data[i])
    return data

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
