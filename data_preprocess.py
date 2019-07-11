import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer, tokenizer_from_json

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
        if os.path.isfile(config.vocab_path):
            with open(config.vocab_path, 'r') as f:
                json_content = f.read()
                self.tokenizer = tokenizer_from_json(json_content)
        else:
            tmp_doc = (self.beg_token + ' ' + self.end_token + ' ') * 100
            docs = [tmp_doc, self.__read_raw_formulas('train')]
            num_tokens = config.vocab_size - 3  # for beg, and, unk token
            self.tokenizer = Tokenizer(num_words=num_tokens,
                                       filters='\t\n', lower=False, oov_token=self.unk_token)
            self.tokenizer.fit_on_texts(docs)
            with open(config.vocab_path, 'w+') as f:
                f.write(self.tokenizer.to_json())

    def get_path(self, mode):
        formulas_path = os.path.join(self.formula_path, '{}_formulas.txt'.format(mode))
        images_folder = os.path.join(self.images_path, 'images_{}'.format(mode))
        return formulas_path, images_folder

    def __read_raw_formulas(self, mode, split=False):
        path = self.get_path(mode)[0]
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

    def start_token(self):
        return self.tokenizer.word_index[self.beg_token]

    def read_formulas(self, mode):
        lines = self.__read_raw_formulas(mode, split=True)
        for i in range(len(lines)):
            lines[i] = '{} {} {}'.format(self.beg_token, lines[i], self.end_token)
        result = self.tokenizer.texts_to_sequences(lines)
        return result

    def read_images(self, mode, index):
        dir_path = self.get_path(mode)[1]
        images_data = []
        for i in index:
            file_path = os.path.join(dir_path, str(i) + '.png')
            if os.path.isfile(file_path):
                image = imageio.imread(file_path)
                images_data.append(image)
        data = np.array(images_data)
        return data

    def decode_formula(self, sequences):
        def normalize(formula):
            start_idx, end_idx = 0, len(formula)
            if formula[:6] == '<BOS> ':
                start_idx = 6
            try:
                end_idx = formula.index(self.end_token)
            except:
                pass
            return formula[start_idx:end_idx]

        sequences_list = sequences.tolist()
        formulas = self.tokenizer.sequences_to_texts(sequences_list)
        formulas = [normalize(formula) for formula in formulas]
        return formulas

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
