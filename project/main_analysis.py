import sys

import numpy as np
from mnist import MNIST

from DNN import DNN
import matplotlib.pyplot as plt

from data_reader import get_mnist_array, vectorize_labels, binarize_image

nb_iter_train = 20
nb_iter_pretrain = 10
lr = 0.1
mini_batch_size = 500

images_train, labels_train, images_test, labels_test = get_mnist_array(shuffle=True)

images_train = binarize_image(images_train, 127)
images_test = binarize_image(images_test, 127)
labels_train = vectorize_labels(labels_train)
# labels_test = vectorize_labels(labels_test)

error_rates = []
range_neurons = np.array(list(range(2, 5)))
for n_neurons in range_neurons:
    network_shape = [784] + [n_neurons * 100] * 2 + [10]
    dbn = DNN(np.array(network_shape))
    dbn_pretrained = DNN(np.array(network_shape))

    # Pretrain
    print("\nPretrain")
    dbn_pretrained.pretrain(images_train, nb_iter_pretrain, lr, mini_batch_size, verbose=True)

    # Train
    print("\nPretrained model")
    dbn_pretrained.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, True)

    print("\nNot pretrained model")
    dbn.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, True)

    error_rates.append((dbn_pretrained.test(images_test, labels_test), dbn.test(images_test, labels_test)))

error_rates = np.array(error_rates)
plt.plot(range_neurons * 100, error_rates[:, 0], label="Pretrained")
plt.plot(range_neurons * 100, error_rates[:, 1], label="Not pretrained")
plt.xlabel("Number of neurons per layers")
plt.ylabel("Error rate")
plt.title("Error rate vs number of neurons per layers")
plt.legend()

plt.savefig("error_rate_vs_number_of_neurons.png")
# plt.show()
