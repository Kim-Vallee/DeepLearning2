from typing import Union

import numpy as np
import tqdm as tqdm


class RBM:
    def __init__(self, p: int, q: int, var: float = 10e-2):
        self.p = p
        self.q = q
        self.w = np.random.normal(0, np.sqrt(var), size=(p, q))
        self.b = np.zeros(q)
        self.a = np.zeros(p)

    def entree_sortie(self, v):
        """
        Train from in to out.

        :param v: matrice m*p avec m la taille du batch (binaire)
        :type v: matrice
        """
        return 1 / (1 + np.exp(-(v @ self.w + self.b)))

    def sortie_entree(self, h):
        """
        Train from out to in.

        :param h: matrice m*q avec m la taille du batch (binaire)
        :type h: matrice
        """
        return 1 / (1 + np.exp(-(h @ self.w.T + self.a)))

    def train(self, X: np.ndarray, nb_iter: int, taille_batch: int, epsilon: float, verbose: bool = True):
        n = X.shape[0]

        if verbose:
            iter_bar = tqdm.tqdm(range(nb_iter))
            batch_bar = tqdm.tqdm(total=len(range(0, n, taille_batch)))
        else:
            iter_bar = range(nb_iter)
            batch_bar = range(0, n, taille_batch)

        for _ in iter_bar:
            np.random.shuffle(X)

            if verbose:
                batch_bar.reset()

            for iter_batch in range(0, n, taille_batch):
                X_batch = X[iter_batch: min(iter_batch + taille_batch, n)]
                t_batch = X_batch.shape[0]
                v_0 = X_batch
                p_h_v0 = self.entree_sortie(v_0)
                h_0 = (np.random.rand(t_batch, self.q) < p_h_v0) * 1
                p_v_h0 = self.sortie_entree(h_0)
                v_1 = (np.random.rand(t_batch, self.p) < p_v_h0) * 1
                p_h_v1 = self.entree_sortie(v_1)
                grad_a = v_0 - v_1
                grad_b = p_h_v0 - p_h_v1
                grad_w = v_0.T @ p_h_v0 - v_1.T @ p_h_v1
                self.a += epsilon * np.sum(grad_a, axis=0) / t_batch
                self.b += epsilon * np.sum(grad_b, axis=0) / t_batch
                self.w += epsilon * grad_w / t_batch
                if verbose:
                    batch_bar.update(1)

            H = self.entree_sortie(X)
            X_rec = self.sortie_entree(H)
            if verbose:
                iter_bar.set_description(f"Reconstruction : {round(np.mean((X - X_rec) ** 2) * 1, 3)}")

        if verbose:
            batch_bar.close()

    def generer_image_RBM(self, nb_iter: int, nb_img: int, tirage_initial: Union[float, np.ndarray] = 0.5) \
            -> np.ndarray:
        v = (np.random.rand(nb_img, self.p) < tirage_initial) * 1
        for j in range(nb_iter):
            h = (np.random.rand(nb_img, self.q) < self.entree_sortie(v)) * 1
            v = (np.random.rand(nb_img, self.p) < self.sortie_entree(h)) * 1
        return v

    def calcul_softmax(self, X: np.ndarray) -> np.ndarray:
        h = X @ self.w + self.b
        return np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True)
