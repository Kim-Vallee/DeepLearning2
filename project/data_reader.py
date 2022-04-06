from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from mnist import MNIST


def lire_alpha_digit(digits: List) -> np.ndarray:
    data = scipy.io.loadmat('data/binaryalphadigs.mat')
    nb_img_per_digit = 39
    img_x, img_y = (20, 16)
    ret_data = np.zeros((nb_img_per_digit * len(digits), img_x * img_y))
    for i, digit in enumerate(digits):
        for j in range(nb_img_per_digit):
            ret_data[i * nb_img_per_digit + j, :] = data['dat'][digit][j].flatten()
    return ret_data


def show_alpha_digits_images(imgs):
    for img in imgs:
        img = img.reshape(20, 16)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()


def get_mnist_array(shuffle: bool = True, data_size: int = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mndata = MNIST('data/MNIST')
    images_train, labels_train = mndata.load_training()
    images_train = np.array(images_train)
    labels_train = np.array(labels_train)
    images_test, labels_test = mndata.load_testing()
    images_test = np.array(images_test)
    labels_test = np.array(labels_test)
    if shuffle:
        seed = np.random.randint(0, 10000)
        for imgs, lbls in [(images_train, labels_train), (images_test, labels_test)]:
            np.random.seed(seed)
            np.random.shuffle(imgs)
            np.random.seed(seed)
            np.random.shuffle(lbls)
    if data_size is not None:
        images_train = images_train[:data_size]
        labels_train = labels_train[:data_size]
        images_test = images_test[:data_size]
        labels_test = labels_test[:data_size]
    return images_train, labels_train, images_test, labels_test


def binarize_image(img: np.ndarray, threshold: float) -> np.ndarray:
    return (img > threshold) * 1


def vectorize_labels(labels: np.ndarray) -> np.ndarray:
    nb_labels = len(np.unique(labels))
    return np.eye(nb_labels)[labels]


if __name__ == '__main__':
    ret = lire_alpha_digit([1])
    print(ret)
