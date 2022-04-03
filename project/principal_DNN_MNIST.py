# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 27/03/2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import numpy as np
from mnist import MNIST

from project.DNN import DNN

network_size = [784, 320, 160, 100, 50, 10]
nb_iter_train = 100
nb_iter_generate = 50
lr = 0.1
mini_batch_size = 50
data_size = 500

mndata = MNIST(r'C:\Users\Kim\Documents\Polytechnique\Deep Learning 2\project\data\MNIST')
images_train, labels_train = mndata.load_training()
images_train = np.array(images_train)
labels_train = np.array(labels_train)
images_test, labels_test = mndata.load_testing()
images_test = np.array(images_test)
labels_test = np.array(labels_test)

seed = np.random.randint(0, 10000)
for imgs, lbls in [(images_train, labels_train), (images_test, labels_test)]:
    np.random.seed(seed)
    np.random.shuffle(imgs)
    np.random.seed(seed)
    np.random.shuffle(lbls)

# Cut data to data_size
images_train = images_train[:data_size]
labels_train = labels_train[:data_size]
images_test = images_test[:data_size]
labels_test = labels_test[:data_size]

# Binarize data
images_train = (images_train > 127) * 1
images_test = (images_test > 127) * 1

# Vectorize labels
labels_train = np.eye(10)[labels_train]
labels_test = np.eye(10)[labels_test]

dbn = DNN(np.array(network_size))
dbn.pretrain(images_train, nb_iter_train, lr, mini_batch_size)
dbn.retropropagation(images_train, labels_train, nb_iter_train, lr, mini_batch_size, True)
error_rate = dbn.test(images_test, labels_test)
print(error_rate)
