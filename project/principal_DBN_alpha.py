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

from DNN import DNN
from data_reader import lire_alpha_digit, show_alpha_digits_images

network_size = [320, 160, 100, 50, 10]
nb_iter_train = 1000
nb_iter_generate = 200
lr = 0.05
mini_batch_size = 20
data_size = 1000

digits = lire_alpha_digit(list(range(10)))
np.random.shuffle(digits)
digits = digits[:data_size]

# region RBM test
# rbm = RBM(network_size[0], network_size[-1])
# rbm.train(digits, nb_iter, mini_batch_size, lr, verbose=True)
#
# imgs = rbm.generer_image_RBM(nb_iter, 10)
# for img in imgs:
#     img = img.reshape(20, 16)
#     plt.imshow(img, cmap=plt.get_cmap('gray'))
#     plt.show()
# endregion

# region DBN test
dbn = DNN(np.array(network_size))
dbn2 = DNN(np.array(network_size))
dbn.pretrain(digits, nb_iter_train, lr, mini_batch_size)

imgs = dbn.generer_image(nb_iter_generate, 10)
imgs2 = dbn2.generer_image(nb_iter_generate, 10)
show_alpha_digits_images(imgs)
show_alpha_digits_images(imgs2)
# endregion
