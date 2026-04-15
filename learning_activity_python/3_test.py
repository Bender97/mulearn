# Copyright (c) Daniel Fusaro. All rights reserved.

from utils.model_utils import read_model
from utils.metrics import accuracy, get_confusion_matrix
from utils.general_functions import describe, print_confusion_matrix
from dataset import read_data, split_data
from model import SimpleNN

colors = ["background", "green", "red", "blue"]

######### PARAMETERS #########
INPUT_SIZE = 3
NUM_CLASSES = len(colors)
MODELNAME= "/flash/simple_nn_model_new.ckpt"
TRAIN_SIZE, VAL_SIZE = 0.5, 0.2
##############################
files= ["/flash/{:s}.csv".format(color) for color in colors]

def test():
    # load the dataset
    all_data, all_labels = read_data(files)

    # split the dataset into train, validation and test
    (_, _), (_, _), (X_test, y_test) = \
            split_data(all_data, all_labels, TRAIN_SIZE, VAL_SIZE, test=True)

    del all_data, all_labels
    #describe(y_test, "test", NUM_CLASSES)


    # load the model
    model = SimpleNN([INPUT_SIZE, 8, 16, NUM_CLASSES])
    read_model(model, MODELNAME)
    print("Model loaded from", MODELNAME)

    # evaluate model performance
    print("Evaluating model performance...")

    acc_test= accuracy(model, X_test, y_test)
    print("Test accuracy: {:.2f} %".format(acc_test * 100))

    confusion_matrix = get_confusion_matrix(model, X_test, y_test, NUM_CLASSES)
    print_confusion_matrix(confusion_matrix)
    
test()