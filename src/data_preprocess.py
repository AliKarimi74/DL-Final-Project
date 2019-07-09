import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer

from config import config


class DataHandler:

    def __init__(self):
        self.formula_path = os.path.join(config.dataset_path, 'formulas')
        self.images_path = os.path.join(config.dataset_path, 'images')

        self.beg_token = '<BOS>'
        self.end_token = '<EOS>'
        self.unk_token = '<UNK>'
        self.tokenizer = None

        self.__fit_tokenizer()

    def __fit_tokenizer(self):
        print('Start fitting tokenizer')
        tmp_doc = (self.beg_token + ' ' + self.end_token + ' ') * 100
        docs = [tmp_doc, self.__read_raw_formulas('train')]
        self.tokenizer = Tokenizer(num_words=config.max_number_of_tokens,
                                   filters='\t\n', lower=False, oov_token=self.unk_token)
        self.tokenizer.fit_on_texts(docs)
        print('Fitting tokenizer finished')

    def __get_path(self, mode):
        formulas_path = os.path.join(self.formula_path, '{}_formulas.txt'.format(mode))
        images_folder = os.path.join(self.images_path, 'images_{}'.format(mode))
        return formulas_path, images_folder

    def __read_raw_formulas(self, mode, split=False):
        path = self.__get_path(mode)[0]
        with open(path, 'r') as f:
            content = f.read()
            if split:
                lines = content.split('\n')
                if not lines[-1]:
                    lines = lines[:-1]
                return lines
            return content

    def pad_token(self):
        return self.tokenizer.word_index[self.end_token]

    def read_formulas(self, mode):
        lines = self.__read_raw_formulas(mode, split=True)
        for i in range(len(lines)):
            lines[i] = '{} {} {}'.format(self.beg_token, lines[i], self.end_token)
        result = self.tokenizer.texts_to_sequences(lines)
        return result

    def read_images(self, mode, index):
        dir_path = self.__get_path(mode)[1]
        images_data = []
        for i in index:
            file_path = os.path.join(dir_path, str(i) + '.png')
            if os.path.isfile(file_path):
                image = imageio.imread(file_path)
                images_data.append(image)
        data = np.array(images_data)
        return data

    def decode_formula(self, sequence):
        formula = self.tokenizer.sequences_to_texts([sequence])[0]
        start_idx, end_idx = 0, len(formula)
        try:
            start_idx = formula.index(self.beg_token)
            start_idx += len(self.beg_token)
        except:
            pass
        try:
            end_idx = formula.index(self.end_token)
        except:
            pass
        return formula[start_idx+1:end_idx]

    def plot_sample_sizes(self):
        lines = self.__read_raw_formulas('train', split=True)
        training_size = len(lines)
        lines += self.__read_raw_formulas('validation', split=True)
        validation_size = len(lines) - training_size
        print('Training set size: ', training_size)
        print('Validation set size: ', validation_size)

        sample_sizes = []
        for l in lines:
            sample_sizes += [len(l)]

        # the histogram of the data
        n, bins, patches = plt.hist(sample_sizes, 20, facecolor='g', alpha=0.75)
        plt.xlabel('length of formula')
        plt.ylabel('sample size')
        plt.title('Histogram of Length of formulas')
        plt.grid(True)
        plt.show()
