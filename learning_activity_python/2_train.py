# Copyright (c) Daniel Fusaro 2026. All rights reserved.

from utils.model_utils import save_model, read_model
from utils.metrics import accuracy_and_loss, get_confusion_matrix
from utils.general_functions import describe, print_confusion_matrix
from dataset import read_data, split_data
from model import SimpleNN

colors = ["background", "green", "red", "blue"]

######### PARAMETERS #########
INPUT_SIZE= 3                   # input size that the model is receiving (for RGB it is 3)

NUM_CLASSES= len(colors)        # number of colors that the model will learn

INITIAL_LR= 0.001                # the higher, the faster the model learns, but less "attentively"
                                # keep it smaller than 1, but greater than di 0.0001

EPOCHS_NUM= 50                  # max number of epochs that we are going to train the model for

TRAIN_SIZE, VAL_SIZE = 0.5, 0.2 # with 0.5 and 0.2 we say we're going to use 50% of the data to train,
                                # 20 % to validate and the remaining (30%) to test the model

MODELNAME    = "/flash/simple_nn_model_new.ckpt" # we will save here the trained model
##############################

files= ["/flash/{:s}.csv".format(color) for color in colors]

def train():

    # load the dataset
    all_data, all_labels = read_data(files)

    # split the dataset in train and validation
    (X_train, y_train), (X_val, y_val) = split_data(all_data, all_labels, TRAIN_SIZE, VAL_SIZE)

    # free unused memory
    del all_data, all_labels

    print("Train\tValid\tNumC\tInitial LR")
    print("{:d}\t\t{:d}\t\t{:d}\t\t{:.3f}".format(len(X_train), len(X_val), NUM_CLASSES, INITIAL_LR))

    describe(y_train, "train", NUM_CLASSES)
    describe(y_val,   "valid", NUM_CLASSES)

    # create the neural network
    model = SimpleNN([INPUT_SIZE, 8, 16, NUM_CLASSES])

    # initialize some useful variables
    best_val = 10^100
    learning_rate = INITIAL_LR

    # train the model
    for epoch in range(EPOCHS_NUM):
        train_loss = 0
        for i in range(len(X_train)):
            model.forward(X_train[i])
            model.backward(y_train[i], learning_rate)

        train_acc, train_loss = accuracy_and_loss(model, X_train, y_train)
        val_acc,val_loss= accuracy_and_loss(model, X_val, y_val)

        msg = " "
        if val_loss < best_val:
            best_val = val_loss
            save_model(model, MODELNAME, colors)
            msg = "*"

        print("Epoch {:d} | Train Loss: {:.4f} | Valid Loss: {:.4f} {:s} | Train Acc: {:.2f} | Val Acc: {:.2f}"
            .format(epoch + 1, train_loss , val_loss, msg, train_acc, val_acc))

        if val_acc >= 100.0:
            print("Early stopping: we reached 100 % accuracy in validation.")
            break

        # at the next epoch we will learn more 'attentively'
        learning_rate *= 0.999


    # measure the performance of the best model
    read_model(model, MODELNAME)
    final_val_acc, _ = accuracy_and_loss(model, X_val, y_val)
    print("Final model accuracy:", final_val_acc)

    confusion_matrix = get_confusion_matrix(model, X_val, y_val, NUM_CLASSES)
    print_confusion_matrix(confusion_matrix)

train()