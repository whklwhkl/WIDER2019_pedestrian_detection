from sklearn.decomposition import IncrementalPCA
from dataset import TrainingSet, ValidationSet, TestSet
from tqdm import trange
import numpy as np
import cv2


if __name__ == '__main__':
    tr = TrainingSet()
    ipca = IncrementalPCA(3)
    x2 = IncrementalPCA(1)
    for i in trange(len(tr)):
        img = cv2.imread(tr[i][0])/255
        pixels = np.reshape(img, [-1,3])
        ipca.partial_fit(pixels)
        x2.partial_fit(pixels ** 2)
    avg = ipca.mean_
    var = x2.mean_ - ipca.mean_ ** 2
    std = var ** .5
    eig_vec = ipca.components_
    eig_val = ipca.explained_variance_
    for n, i in zip(['mean', 'std', 'self._eig_vec', 'self._eig_val'],[avg, var, eig_vec, eig_val]):
        print(f'{n} = np.array({i})')
