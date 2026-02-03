# Copyright (c) Daniel Fusaro. All rights reserved.
import struct, os

def save_model(model, filename, classes):
    with open(filename, "wb") as f:
        for layer in [model.fc1, model.fc2, model.fc3]:
            for i in range(layer.out_f):
                for j in range(layer.in_f):
                    f.write(bytearray(struct.pack("f", layer.W[i][j])))
            for i in range(layer.out_f):
                f.write(bytearray(struct.pack("f", layer.b[i])))
        # classes is list of strings
        for i in range(len(classes)):
            classes[i] = classes[i].replace("\n", "").strip()
        s = "".join([c + "\n" for c in classes])
        f.write(s.encode("utf-8"))


def read_model(model, filename):
    with open(filename, "rb") as f:
        for layer in [model.fc1, model.fc2, model.fc3]:
            for i in range(layer.out_f):
                for j in range(layer.in_f):
                    bytes_read = f.read(4)
                    layer.W[i][j] = struct.unpack("f", bytes_read)[0]
            for i in range(layer.out_f):
                bytes_read = f.read(4)
                layer.b[i] = struct.unpack("f", bytes_read)[0]
        # Read classes
        classes = []
        class_data = f.read().decode("utf-8")
        for line in class_data.strip().split("\n"):
            # remove trailing newline and spaces
            line = line.replace("\n", "").strip()
            if line != "":
                classes.append(line)
        return classes
                