import os
import numpy as np

from configuration import config, h_params
from data.data_preprocess import DataHandler


class DataGenerator:

    def __init__(self, mode):
        self.data = DataHandler()
        self.pad_token = self.data.pad_token()
        self.mode = mode

    def __fetch_formulas(self):
        self.formulas = self.data.read_formulas(self.mode)
        self.sorted_index = sorted(range(len(self.formulas)), key=lambda k: len(self.formulas[k]))

        if len(self.formulas) == 0:
            images_path = self.data.get_path(self.mode)[1]
            images_count = 0
            while os.path.isfile(os.path.join(images_path, str(images_count)+'.png')):
                images_count += 1
            self.sorted_index = range(images_count)

    def __get_formula(self, index):
        res, sizes = [], []
        if len(self.formulas) > 0:
            res = [self.formulas[idx] for idx in index]
            sizes = [len(f) for f in res]
            max_len = config.max_generate_steps
            for i in range(len(res)):
                size = len(res[i])
                remain = max_len - size
                if remain >= 0:
                    res[i] = res[i] + [self.pad_token]*remain
                else:
                    res[i] = res[i][:max_len]
        # 0 index is reserved in tokenizer, so all elements are greater than zero
        res = np.array(res)
        res -= 1
        return res, sizes

    def decode_formulas(self, sequences):
        sequences = np.copy(sequences)
        sequences += 1
        return self.data.decode_formula(sequences)

    def generator(self, n_epoch, percentage_limit=None):
        self.__fetch_formulas()
        batch_size = h_params.batch_size
        epoch = 0
        head = 0
        while epoch < n_epoch:
            index = self.sorted_index[head:head + batch_size]
            images = self.data.read_images(self.mode, index)
            formulas, formulas_len = self.__get_formula(index)
            head += batch_size
            percentages = head / len(self.sorted_index)
            yield epoch, percentages, images, formulas, formulas_len

            if head >= len(self.sorted_index)\
                    or percentage_limit is not None and percentages >= percentage_limit:
                head = 0
                epoch += 1
