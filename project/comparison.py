# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 12/04/2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from tensorflow import keras

from DNN import DNN
from RBM import RBM
from VAE import VAE
from data_reader import get_mnist_array

network_size = [784, 200, 200, 30, 10]
nb_iter_train = 100
nb_iter_generate = 50
lr = 0.1
mini_batch_size = 100

images_train, labels_train, images_test, labels_test = get_mnist_array(shuffle=True)

# Binarize data
images_train = (images_train > 127) * 1
images_test = (images_test > 127) * 1

# Vectorize labels
labels_train = np.eye(10)[labels_train]
labels_test = np.eye(10)[labels_test]

# Prepare the models
rbm = RBM(network_size[0], network_size[-1])
dbn = DNN(np.array(network_size))
vae = VAE()  # https://keras.io/examples/generative/vae/
vae.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.0), loss=keras.losses.CategoricalCrossentropy(),
            verbose=0)

# Train the models
rbm.train(images_train, nb_iter_train, mini_batch_size, lr)
dbn.pretrain(images_train, nb_iter_train, lr, mini_batch_size)
vae.fit(images_train.reshape(-1, 28, 28), epochs=nb_iter_train, batch_size=mini_batch_size)

# Generate images
nb_imgs = 10
imgs_rbm = rbm.generer_image_RBM(nb_iter_generate, nb_imgs)
imgs_dbn = dbn.generer_image(nb_iter_generate, nb_imgs)
imgs_vae = []
for i in range(nb_imgs):
    imgs_vae.append(vae.decoder.predict(np.eye(10)[i].flatten())[0].reshape(28, 28))

imgs_vae = np.array(imgs_vae)

# Save the images
np.savez('images_rbm.npz', imgs_rbm=imgs_rbm, imgs_dbn=imgs_dbn, imgs_vae=imgs_vae)