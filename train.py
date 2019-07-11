import os
import tensorflow as tf

from config import config
from data_generator import DataGenerator
from model.image_to_latex import ImageToLatexModel

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_string('model_name', 'image2latex', 'Model name')


def main(args):
    data_gen = DataGenerator('train')
    start_token, pad_token = data_gen.data.start_token(), data_gen.data.pad_token()

    model_dir = os.path.join('runs', FLAGS.model_name)

    print('Start building graph')
    tf.reset_default_graph()
    model = ImageToLatexModel(start_token, pad_token)
    print('Graph building finished!')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Start fitting ...')
    for epoch, percentage, images, formulas, _ in data_gen.generator(config.n_epochs):
        loss, step = model.train_step(sess, images, formulas)
        print('loss:', loss)
        print('step:', step)


if __name__ == '__main__':
    tf.app.run()
