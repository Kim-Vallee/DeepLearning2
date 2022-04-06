import numpy as np
from mnist import MNIST

from project.DNN import DNN
import matplotlib.pyplot as plt

from project.data_reader import get_mnist_array, vectorize_labels, binarize_image

nb_iter_train = 100
lr = 0.1
mini_batch_size = 1000

images_train, labels_train, images_test, labels_test = get_mnist_array(shuffle=True)

images_train = binarize_image(images_train, 127)
images_test = binarize_image(images_test, 127)
labels_train = vectorize_labels(labels_train)
labels_test = vectorize_labels(labels_test)

error_rates = []
range_couches = range(2, 6)
for n_couches in range_couches:
    network_shape = [784] + n_couches * [200] + [10]
    dbn = DNN(np.array(network_shape))
    dbn_pretrained = DNN(np.array(network_shape))

    # Pretrain
    print("Pretrain")
    dbn_pretrained.pretrain(images_train, nb_iter_train, lr, mini_batch_size)

    # Train
    print("Pretrained model")
    dbn_pretrained.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, True)

    print("Not pretrained model")
    dbn.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, True)

    error_rates.append((dbn_pretrained.test(images_test, labels_test), dbn.test(images_test, labels_test)))

error_rates = np.array(error_rates)
print("wait")
plt.plot(range_couches, error_rates[:, 0], label="Pretrained")
plt.plot(range_couches, error_rates[:, 1], label="Not pretrained")
plt.xlabel("Number of layers")
plt.ylabel("Error rate")
plt.title("Error rate vs number of layers")
plt.legend()

plt.savefig("error_rate_vs_number_of_layers.png")
plt.show()
