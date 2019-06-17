from sklearn.decomposition import IncrementalPCA
from dataset import TrainingSet, ValidationSet, TestSet
from tqdm import trange
import numpy as np
import cv2


def get_stat(dataset):
    """
        calculate [mean, std, eig_vec, eig_val] of the given dataset
    """
    ipca = IncrementalPCA(3)
    x2 = IncrementalPCA(1)
    for i in trange(len(dataset)):
        img = cv2.imread(dataset[i][0])/255
        pixels = np.reshape(img, [-1,3])
        ipca.partial_fit(pixels)
        x2.partial_fit(pixels ** 2)
    avg = ipca.mean_
    var = x2.mean_ - avg ** 2
    std = var ** .5
    return avg, std, ipca.components_, ipca.explained_variance_


if __name__ == '__main__':
    tr = TrainingSet()

    avg, std, eig_vec, eig_val = get_stat(tr)

    for n, i in zip(['mean', 'std', 'self._eig_vec', 'self._eig_val'],[avg, std, eig_vec, eig_val]):
        print(f'{n} = np.array({i.tolist()})')
