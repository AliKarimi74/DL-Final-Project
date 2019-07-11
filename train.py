import os
import logging
import tensorflow as tf
import numpy as np

from config import config
from data_generator import DataGenerator
from model.image_to_latex import ImageToLatexModel
from eval.evaluation import evaluation

# mute tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_string('model_name', 'image2latex', 'Model name')
flags.DEFINE_boolean('check_on_small_data', False, 'Train model on 1% of data to figure out model can overfit or not')


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

    loss_history = []
    n_epochs, per_limit = (1, 0.01) if FLAGS.check_on_small_data else (config.n_epochs, None)
    validation_set = 'train' if FLAGS.check_on_small_data else 'validation'

    log_every = config.log_every
    eval_every_epoch = config.eval_every_epoch
    log_template = 'Epoch {}({}), step = {} => Loss avg: {}'

    log('Start fitting ...')

    for epoch, percentage, images, formulas, _ in train_set.generator(n_epochs, per_limit):
        loss, step = model.train_step(sess, images, formulas)
        loss_history += [loss]
        s = step + 1

        if s % log_every == 0:
            percent = '{0:2.2f}%'.format(100 * percentage)
            loss_average = np.mean(np.array(loss_history))
            log(log_template.format(epoch + 1, percent, s, loss_average))
            loss_history = []

        if epoch % eval_every_epoch == 0 and percentage >= 1:
            exact_match, bleu, edit_distance = evaluation(session=sess, model=model,
                                                          mode=validation_set, percent_limit=per_limit)


if __name__ == '__main__':
    tf.app.run()
