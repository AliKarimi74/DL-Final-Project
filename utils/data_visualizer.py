import random
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator

gen = DataGenerator('train')

for ep, per, images, formulas, l in gen.generator(1):
    for i in range(1):
        idx = random.randint(0, images.shape[0] - 1)
        img = images[idx]
        formula = np.expand_dims(formulas[idx], axis=0)
        formula = gen.decode_formulas(formula)[0]
        print(formula)

        plt.imshow(img, cmap='gray')
        plt.title(formula)
        plt.show()
