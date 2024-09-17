from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def load_single_image(img_path):
    # Extract ground truth labels (Y) from the file name
    Y = img_path[img_path.rfind('_') + 1:-4]

    images = Image.open(img_path)
    img_mat = np.asarray(images)

    # Calculate dimensions and number of pixels
    height = int(img_mat.shape[0])
    width = int(img_mat.shape[1] / len(Y))
    n_pixels = height * width

    X = np.zeros([int(n_pixels + n_pixels * (n_pixels - 1) / 2), len(Y)])
    for i in range(len(Y)):
        # single letter i
        letter = img_mat[:, i * width:(i + 1) * width] / 255
        # compute features
        x = letter.flatten()
        # Copy features to X and update cnt
        X[0:len(x), i] = x
        cnt = n_pixels
        for j in range(0, n_pixels - 1):
            for k in range(j + 1, n_pixels):
                X[cnt, i] = x[j] * x[k]
                cnt = cnt + 1
        # Normalize the feature vector
        X[:, i] = np.array(X[:, i] / np.linalg.norm(X[:, i]))

    X = X.T
    return X, Y


def load_images(image_folder, description):
    X = []
    Y = []
    for file in tqdm(listdir(image_folder), desc=description):
        if file.endswith(f'.png'):
            path = join(image_folder, file)
            if isfile(path):
                X_, Y_ = load_single_image(path)
                X.append(X_)
                Y.append(Y_)

    return X, Y
