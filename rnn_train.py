import argparse
import numpy as np
import tensorflow as tf
import tflearn
import os
import matplotlib.pyplot as plt
from rnn_utils import get_network_wide, get_data

def load_labels(label_file):
    label = {}
    count = 0
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_asascii_lines:
        label[l.strip()] = count
        count += 1
    return label

def train_rnn(input_data_dump, num_frames_per_video, batch_size, labels, model_file, epochs=10):
    # Get our data.
    X_train, X_test, y_train, y_test = get_data(input_data_dump, num_frames_per_video, labels, True)

    num_classes = len(labels)
    size_of_each_frame = X_train.shape[2]

    # Get our network.
    net = get_network_wide(num_frames_per_video, size_of_each_frame, num_classes)

    # Train the model.
    try:
        model = tflearn.DNN(net, tensorboard_verbose=0)
        model.load('checkpoints/' + model_file)
        print("\nModel already exists! Loading it")
        print("Model Loaded")
    except Exception:
        model = tflearn.DNN(net, tensorboard_verbose=0)
        print("\nNo previous checkpoints of %s exist" % (model_file))

    model.fit(X_train, y_train, validation_set=(X_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100,
              n_epoch=epochs)

    # Save it.
    x = input("Do you want to save the model and overwrite? y or n: ")
    if(x.strip().lower() == "y"):
        model.save('checkpoints/' + model_file)

    return model

def test_rnn(input_data_dump, num_frames_per_video, labels, model_file, batch_size=32):
    # Get our data.
    X, Y = get_data(input_data_dump, num_frames_per_video, labels, False)

    num_classes = len(labels)
    size_of_each_frame = X.shape[2]

    # Get our network.
    net = get_network_wide(num_frames_per_video, size_of_each_frame, num_classes)

    # Load the model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load('checkpoints/' + model_file)

    predictions = model.predict(X)
    predictions = np.array([np.argmax(pred) for pred in predictions])
    Y = np.array([np.argmax(each) for each in Y])

    # Writing predictions and gold labels to file
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    with open("result.txt", "w") as f:
        f.write("gold, pred\n")
        for a, b in zip(Y, predictions):
            f.write("%s %s\n" % (rev_labels[a], rev_labels[b]))

    acc = 100 * np.sum(predictions == Y) / len(Y)
    print("Accuracy: ", acc)

    # Plot the accuracy curve.
    plt.plot(Y, label="True Labels")
    plt.plot(predictions, label="Predictions")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a RNN')
    parser.add_argument("input_file_dump", help="file containing intermediate representation of gestures from inception model")
    parser.add_argument("model_file", help="Name of the model file to be dumped. Model file is created inside a checkpoints
