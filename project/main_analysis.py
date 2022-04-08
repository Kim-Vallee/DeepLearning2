import sys

import numpy as np
from mnist import MNIST

from DNN import DNN
# import matplotlib.pyplot as plt

from data_reader import get_mnist_array, vectorize_labels, binarize_image
import pickle

nb_iter_train = 100
nb_iter_pretrain = 200
lr = 0.1
mini_batch_size = 500

images_train, labels_train, images_test, labels_test = get_mnist_array(shuffle=True)

images_train = binarize_image(images_train, 127)
images_test = binarize_image(images_test, 127)
labels_train = vectorize_labels(labels_train)
# labels_test = vectorize_labels(labels_test)

def nb_neurons_per_layer():
    error_rates = []
    range_neurons = np.array(list(range(2, 5)))
    for n_neurons in range_neurons:
        network_shape = [784] + [n_neurons * 100] * 2 + [10]
        dbn = DNN(np.array(network_shape))
        dbn_pretrained = DNN(np.array(network_shape))

        # Pretrain
        # print("\nPretrain")
        dbn_pretrained.pretrain(images_train, nb_iter_pretrain, lr, mini_batch_size, verbose=False)

        # Train
        # print("\nPretrained model")
        dbn_pretrained.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, False)

        # print("\nNot pretrained model")
        dbn.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, False)

        error_rates.append((dbn_pretrained.test(images_test, labels_test), dbn.test(images_test, labels_test)))

    error_rates = np.array(error_rates)
    # plt.plot(range_neurons * 100, error_rates[:, 0], label="Pretrained")
    # plt.plot(range_neurons * 100, error_rates[:, 1], label="Not pretrained")
    # plt.xlabel("Number of neurons per layers")
    # plt.ylabel("Error rate")
    # plt.title("Error rate vs number of neurons per layers")
    # plt.legend()

    # plt.savefig("error_rate_vs_number_of_neurons.png")
    with open('error_rate_vs_number_of_neurons.pickle', "wb+") as f:
        pickle.dump([range_neurons*100, error_rates, "Number of neurons per layers", "Error rate", "Error rate vs number of neurons per layers"], f)

def nb_layers():
    error_rates = []
    range_couches = np.array(list(range(2, 5)))
    for n_couches in range_couches:
        network_shape = [784] + [200] * n_couches + [10]
        dbn = DNN(np.array(network_shape))
        dbn_pretrained = DNN(np.array(network_shape))

        # Pretrain
        # print("\nPretrain")
        dbn_pretrained.pretrain(images_train, nb_iter_pretrain, lr, mini_batch_size, verbose=False)

        # Train
        # print("\nPretrained model")
        dbn_pretrained.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, False)

        # print("\nNot pretrained model")
        dbn.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, False)

        error_rates.append((dbn_pretrained.test(images_test, labels_test), dbn.test(images_test, labels_test)))

    error_rates = np.array(error_rates)
    # plt.plot(range_neurons, error_rates[:, 0], label="Pretrained")
    # plt.plot(range_neurons, error_rates[:, 1], label="Not pretrained")
    # plt.xlabel("Number of layers")
    # plt.ylabel("Error rate")
    # plt.title("Error rate vs number of layers")
    # plt.legend()

    # plt.savefig("error_rate_vs_number_of_layers.png")
    with open('error_rate_vs_number_of_layers.pickle', "wb+") as f:
        pickle.dump([range_couches, error_rates, "Number of layers", "Error rate", "Error rate vs number of layers"], f)

def nb_train_data():
    error_rates = []
    range_data = [1000, 3000, 7000, 10000, 30000, 60000]
    for n_data in range_data:
        network_shape = [784] + [200] * 3 + [10]
        dbn = DNN(np.array(network_shape))
        dbn_pretrained = DNN(np.array(network_shape))

        train_data = images_train[:n_data]
        train_labels = labels_train[:n_data]

        # Pretrain
        # print("\nPretrain")
        dbn_pretrained.pretrain(train_data, nb_iter_pretrain, lr, mini_batch_size, verbose=False)

        # Train
        # print("\nPretrained model")
        dbn_pretrained.retropropagation(train_data, train_labels, nb_iter_train, lr, mini_batch_size, False)

        # print("\nNot pretrained model")
        dbn.retropropagation(train_data, train_labels, nb_iter_train, lr, mini_batch_size, False)

        error_rates.append((dbn_pretrained.test(images_test, labels_test), dbn.test(images_test, labels_test)))

    error_rates = np.array(error_rates)
    # plt.plot(range_data, error_rates[:, 0], label="Pretrained")
    # plt.plot(range_data, error_rates[:, 1], label="Not pretrained")
    # plt.xlabel("Number of data")
    # plt.ylabel("Error rate")
    # plt.title("Error rate vs number of data")
    # plt.legend()

    # plt.savefig("error_rate_vs_number_of_data.png")
    with open('error_rate_vs_number_of_data.pickle', "wb+") as f:
        pickle.dump([range_data, error_rates, "Number of data", "Error rate", "Error rate vs number of data"], f)
    
if __name__ == "__main__":
    nb_neurons_per_layer()
    nb_layers()
    nb_train_data()