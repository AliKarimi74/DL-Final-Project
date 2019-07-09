import numpy as np

from config import config
from hyperparams import h_params
from data_preprocess import DataHandler


class DataGenerator:

    def __init__(self, mode):
        self.data = DataHandler()
        self.pad_token = self.data.pad_token()
        self.mode = mode

    def __fetch_formulas(self):
        self.formulas = self.data.read_formulas(self.mode)
        self.sorted_index = sorted(range(len(self.formulas)), key=lambda k: len(self.formulas[k]))

    def __get_formula(self, index):
        res = [self.formulas[idx] for idx in index]
        max_len = len(res[-1])
        for i in range(len(res)):
            size = len(res[i])
            remain = max_len - size
            res[i] = res[i] + [self.pad_token]*remain
        # 0 index is reserved in tokenizer, so all elements are greater than zero
        res = np.array(res)
        res -= 1
        return res

    def generator(self, n_epoch):
        self.__fetch_formulas()
        batch_size = h_params.batch_size
        epoch = 0
        head = 0
        while epoch < n_epoch:
            index = self.sorted_index[head:head + batch_size]
            images = self.data.read_images(self.mode, index)
            targets = self.__get_formula(index)
            percentages = head / len(self.sorted_index)
            yield epoch, percentages, images, targets

            head += batch_size
            if head >= len(self.sorted_index):
                head = 0
                epoch += 1
