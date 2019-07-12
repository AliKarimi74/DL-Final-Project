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
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('model_name', 'image2latex', 'Model name')
flags.DEFINE_boolean('load_from_previous', False, 'Whether to load from previous session or not')
flags.DEFINE_boolean('check_on_small_data', False, 'Train model on 1% of data to figure out model can overfit or not')
FLAGS = flags.FLAGS


def main(args):
    def log(msg, new_section=False):
        logging.info(msg)
        if new_section:
            logging.info('------------------------------------')

    train_set = DataGenerator('train')
    start_token, pad_token = train_set.data.start_token(), train_set.data.pad_token()

    model_path = os.path.join('runs', FLAGS.model_name + '.ckpt')

    gpu_is_available = tf.test.is_gpu_available()
    log('GPU is available: ' + str(gpu_is_available))
    log('Start building graph')
    tf.reset_default_graph()
    model = ImageToLatexModel(start_token, pad_token)
    log('Graph building finished!', True)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if FLAGS.load_from_previous:
        saver.restore(sess, model_path)

    loss_history = []
    small_data = FLAGS.check_on_small_data
    n_epochs, per_limit = (500, 0.01) if small_data else (config.n_epochs, None)
    validation_set = 'train' if small_data else 'validation'

    log_every = config.log_every if gpu_is_available else 1
    eval_every_epoch = config.eval_every_epoch if not small_data else n_epochs // 20
    log_template = 'Epoch {}({}), step = {} => Loss avg: {}'

    log('Start fitting ' + ('on small data' if small_data else '...'))

    for epoch, percentage, images, formulas, _ in train_set.generator(n_epochs, per_limit):
        loss, step, first_cnn_filter = model.train_step(sess, images, formulas)
        loss_history += [loss]

        percentage_condition = percentage >= 1 or (per_limit is not None and percentage > per_limit)
        if step % log_every == 0 or percentage_condition:
            percent = '{0:2.2f}%'.format(100 * percentage)
            loss_average = np.mean(np.array(loss_history))
            log(log_template.format(epoch + 1, percent, step, loss_average))
            loss_history = []

        if epoch % eval_every_epoch == 0 and percentage_condition:
            exact_match, bleu, edit_distance = evaluation(session=sess, model=model,
                                                          mode=validation_set, percent_limit=per_limit)
            if not small_data:
                saver.save(sess, model_path, step)

    evaluation(session=sess, model=model, mode=validation_set, percent_limit=per_limit)
    if not small_data:
        saver.save(sess, model_path, step)


if __name__ == '__main__':
    tf.app.run()
