from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


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


if __name__ == '__main__':
    ret = lire_alpha_digit([1])
    print(ret)
