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
from typing import List, Tuple

import numpy as np

from RBM import RBM


class DNN:
    def __init__(self, network_shape: np.ndarray):
        """
        Initialize the DNN

        :param network_shape: the shape of the network starting from layer 0 to layer P.
        :type network_shape: np.ndarray
        """
        self._DNN: List[RBM] = [None] * (network_shape.size - 1)
        for i in range(network_shape.size - 1):
            self._DNN[i] = RBM(network_shape[i], network_shape[i + 1])

    def pretrain(self, X: np.ndarray, nb_iter: int, learning_rate: float, minibatch_size: int):
        """
        Pretrain the DNN.

        :param X: Data to pretrain the DNN.
        :type X: np.ndarray
        :param nb_iter: Number of iterations of gradient descent.
        :type nb_iter: int
        :param learning_rate: Learning rate of the gradient descent.
        :type learning_rate: float
        :param minibatch_size: Size of the minibatch.
        :type minibatch_size: int
        """

        x = X
        for rbm in self._DNN:
            rbm.train(x, nb_iter, minibatch_size, learning_rate, verbose=False)
            x = rbm.entree_sortie(x)

    def generer_image(self, nb_iter: int, nb_img: int) -> np.ndarray:
        """
        Generate image with Gibbs procedure.

        :param nb_iter: Number of iteration of Gibbs.
        :type nb_iter: int
        :param nb_img: Number of images to be generated
        :type nb_img: int
        :return: The images generated
        :rtype: np.ndarray
        """

        h = 0.5
        for rbm in self._DNN[::-1]:
            v = (np.random.rand(nb_img, rbm.q) < h) * 1
            for j in range(nb_iter):
                h = (np.random.rand(nb_img, rbm.p) < rbm.sortie_entree(v)) * 1
                v = (np.random.rand(nb_img, rbm.q) < rbm.entree_sortie(h)) * 1

        return h

    def entree_sortie_reseau(self, X: np.ndarray) -> Tuple[List, np.ndarray]:
        sorties = []
        h = X
        for rbm in self._DNN[:-1]:
            h = rbm.entree_sortie(h)
            sorties.append(h)
        probs = self._DNN[-1].calcul_softmax(h)
        return sorties, probs

    def retropropagation(self, X: np.ndarray, labels: np.ndarray, nb_iter: int, learning_rate: float,
                         minibatch_size: int, verbose: bool = False):

        def cost(true_labels, predicted_labels):
            return -np.sum(true_labels * np.log(predicted_labels) + (1 - true_labels) * np.log(1 - predicted_labels))

        N = X.shape[0]

        for i in range(nb_iter):
            seed = np.random.randint(0, 10000)
            X_copy = X.copy()
            labels_copy = labels.copy()
            np.random.seed(seed)
            np.random.shuffle(X_copy)
            np.random.seed(seed)
            np.random.shuffle(labels_copy)

            # Mini-batch (Should I copy the RBMs?)
            for j in range(0, N, minibatch_size):
                X_minibatch = X_copy[i: min(i + minibatch_size, N)]
                labels_minibatch = labels_copy[i: min(i + minibatch_size, N)]
                batch_size = X_minibatch.shape[0]
                sorties, probs = self.entree_sortie_reseau(X_minibatch)
                estimated_labels = probs

                # Computing the gradient
                dZ, dw, db, dA = [], [], [], []
                dA.append(estimated_labels - labels_minibatch)
                for k in range(len(self._DNN) - 1, -1, -1):
                    dZ.append(dA[-1])  # No activation function
                    if k == 0:
                        dw.append(X_minibatch.T @ dZ[-1] / batch_size)
                    else:
                        dw.append(sorties[k-1].T @ dZ[-1] / batch_size)
                    db.append(np.sum(dZ[-1], axis=0) / batch_size)
                    dA.append(dZ[-1] @ self._DNN[k].w.T)

                # Reverting the gradient
                dw = dw[::-1]
                db = db[::-1]

                # Updating the weights
                for k in range(len(self._DNN)):
                    self._DNN[k].w -= learning_rate * dw[k]
                    self._DNN[k].b -= learning_rate * db[k]

            if verbose:
                _, probs = self.entree_sortie_reseau(X)
                print(cost(labels, probs))  # BCE

    def test(self, X: np.ndarray, labels: np.ndarray) -> float:
        sorties, probs = self.entree_sortie_reseau(X)
        estimated_labels = sorties[-1]
        error_rate = np.sum(estimated_labels != labels) / X.shape[0]
        return error_rate
