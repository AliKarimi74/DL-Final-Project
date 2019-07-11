import os
import logging
import tensorflow as tf
import numpy as np

from config import config
from data_generator import DataGenerator
from model.image_to_latex import ImageToLatexModel
from eval.evaluation import evaluation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_string('model_name', 'image2latex', 'Model name')


def main(args):
    def log(msg, new_section=False):
        logging.info(msg)
        if new_section:
            logging.info('------------------------------------')

    train_set = DataGenerator('train')
    start_token, pad_token = train_set.data.start_token(), train_set.data.pad_token()

    model_dir = os.path.join('runs', FLAGS.model_name)

    log('Start building graph')
    tf.reset_default_graph()
    model = ImageToLatexModel(start_token, pad_token)
    log('Graph building finished!', True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    log_every = config.log_every
    eval_every = config.eval_every
    log_template = 'Epoch {}({}), step = {} => Loss avg: {}'

    log('Start fitting ...')

    loss_history = []
    for epoch, percentage, images, formulas, _ in train_set.generator(config.n_epochs):
        loss, step = model.train_step(sess, images, formulas)
        loss_history += [loss]
        s = step + 1

        if s % log_every == 0:
            percentage = '{0:2.2f}%'.format(100 * percentage)
            loss_average = np.mean(np.array(loss_history))
            log(log_template.format(epoch + 1, percentage, s, loss_average))
            loss_history = []

        if step % eval_every == 0:
            exact_match, bleu, edit_distance = evaluation(session=sess, model=model)


if __name__ == '__main__':
    tf.app.run()
