import numpy as np
from sklearn.datasets import fetch_mldata

class MNISTLoader:
    def __init__(self, dataset_dir='dataset', shuffle=True):
        mnist = fetch_mldata('MNIST original', data_home=dataset_dir)
        self.image, self.label = mnist.data, mnist.target
        self.size = mnist.data.shape[0]
        self.shuffle = shuffle

        self._cur = 0

    def next_batch(self, batch_size):
        if self._cur + batch_size > self.size:
            if self.shuffle:
                self._shuffle()
            self._cur = 0

        image_batch = self.image[self._cur:self._cur + batch_size]
        label_batch = self.label[self._cur:self._cur + batch_size]
        self._cur += batch_size
        return image_batch, label_batch

    def _shuffle(self):
        perm = np.random.permutation(self.size)
        self.image, self.label = self.image[perm], self.label[perm]
