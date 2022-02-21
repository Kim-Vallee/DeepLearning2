import numpy as np
import tqdm.notebook as tqdm
import pandas as pd
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, p: int, q: int, scale: float = 10e-2):
        self.p = p
        self.q = q
        self.w = np.random.normal(0, scale, size=(p, q))
        self.b = np.zeros(q)
        self.a = np.zeros(p)

    def _in_out(self, v):
        """
        Train from in to out.
        :param v: matrice m*p avec m la taille du batch (binaire)
        :type v: matrice
        """
        return 1 / (1 + np.exp(-(v @ self.w + self.b)))

    def _out_in(self, h):
        """
        Train from out to in.
        :param h: matrice m*q avec m la taille du batch (binaire)
        :type h: matrice
        """
        return 1 / (1 + np.exp(-(h @ self.w.T + self.a)))

    def train(self, X: np.ndarray, nb_iter: int, taille_batch: int, epsilon: float):
        n = X.shape[0]
        iter_bar = tqdm.tqdm(range(nb_iter))
        batch_bar = tqdm.tqdm(total=len(range(0, n, taille_batch)))

        for _ in iter_bar:
            np.random.shuffle(X)
            batch_bar.reset()
            for iter_batch in range(0, n, taille_batch):
                X_batch = X[iter_batch: min(iter_batch + taille_batch, n)]
                t_batch = X_batch.shape[0]
                v_0 = X_batch
                p_h_v0 = self._in_out(v_0)
                h_0 = (np.random.rand(t_batch, self.q) < p_h_v0) * 1
                p_v_h0 = self._out_in(h_0)
                v_1 = (np.random.rand(t_batch, self.p) < p_v_h0) * 1
                p_h_v1 = self._in_out(v_1)
                grad_a = v_0 - v_1
                grad_b = p_h_v0 - p_h_v1
                grad_w = v_0.T @ p_h_v0 - v_1.T @ p_h_v1
                self.a += epsilon * np.sum(grad_a, axis=0)
                self.b += epsilon * np.sum(grad_b, axis=0)
                self.w += epsilon * grad_w
                batch_bar.update(1)

            H = self._in_out(X)
            X_rec = self._out_in(H)
            iter_bar.set_description(f"Reconstruction : {np.mean((X - X_rec) ** 2)}")
        batch_bar.close()


if __name__ == "__main__":
    train_data = pd.read_csv("train.csv", delimiter=",")
    labels = train_data["label"].tolist()
    p_input = 784
    q_input = 100
    rbm = RBM(p_input, q_input)
    X_train = train_data.iloc[:, 1:].to_numpy().astype('int')
    print(X_train.shape, X_train)
    rbm.train(X_train, 10, 10, 10e-4)
    outcome = rbm._out_in((np.random.rand(q_input) < 0.5) * 1)
    img = np.zeros((24, 24))
    for i in range(1000):
        v = (np.random.rand(1, 784) < outcome) * 1
        img += v.reshape(24, 24)

    img /= 1000
    plt.imshow(img, cmap="Greys")
    plt.show()
