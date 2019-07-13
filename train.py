import os
import tensorflow as tf
import numpy as np

from config import config
from hyperparams import h_params
from data_generator import DataGenerator
from model.image_to_latex import ImageToLatexModel
from eval.evaluation import evaluation
from utils.logger import logger, log as LOG

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
        LOG(msg, new_section)

    train_set = DataGenerator('train')
    start_token, pad_token = train_set.data.start_token(), train_set.data.pad_token()

    model_name = FLAGS.model_name
    model_path = os.path.join(config.save_path, model_name)
    secondary_model_path = os.path.join(config.secondary_path, config.save_path, model_name)
    logger.set_model_name(model_name)

    log(config, True)
    log(h_params, True)

    gpu_is_available = tf.test.is_gpu_available()
    mini_loss_history, loss_hist = [], []
    small_data = FLAGS.check_on_small_data
    n_epochs, per_limit = (500, 0.01) if small_data else (config.n_epochs, None)
    validation_set = 'train' if small_data else 'validation'

    log_every = config.log_every if gpu_is_available else 2
    eval_every_epoch = config.eval_every_epoch if not small_data else n_epochs // 20
    log_template = 'Epoch {0} ({1}), step = {2} => Loss: {3:1.3f}, lr: {4}'  # , Accuracy: {4:2.2f}'

    step = None
    def run_eval():
        nonlocal sess, model, saver, validation_set, per_limit, model_path, secondary_model_path, step
        evaluation(session=sess, model=model, mode=validation_set, percent_limit=per_limit)
        if not small_data:
            path = saver.save(sess, model_path, step)
            log('Model saved in {}'.format(path), new_section=True)
            try:
                saver.save(sess, secondary_model_path, step)
            except:
                pass

    log('GPU is available: ' + str(gpu_is_available))
    log('Start building graph')
    tf.reset_default_graph()
    model = ImageToLatexModel(start_token, pad_token)
    log('Graph building finished!', True)

    sess = tf.Session()
    init_opt = tf.global_variables_initializer()

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    restored = False
    if FLAGS.load_from_previous:
        try:
            print(saver.last_checkpoints)
            saver.restore(sess, model_path)
            step = sess.run([model.step])[0]
            log('Model restored from last session, current step: {}'.format(step))
            run_eval()
            restored = True
        except:
            log('Can\'t load from previous session')
    if not restored:
        sess.run(init_opt)

    log('Start fitting ' + ('on small data' if small_data else '...'))

    for epoch, percentage, images, formulas, _ in train_set.generator(n_epochs, per_limit):
        loss, step, lr = model.train_step(sess, images, formulas)
        mini_loss_history += [loss]

        percentage_condition = percentage >= 1 or (per_limit is not None and percentage > per_limit)
        if step % log_every == 0 or percentage_condition:
            percent = '{0:2.2f}%'.format(100 * percentage)
            loss_average = np.mean(np.array(mini_loss_history))
            log(log_template.format(epoch + 1, percent, step, loss_average, lr))
            loss_hist += [loss_average]
            mini_loss_history = []

        if epoch % eval_every_epoch == 0 and percentage_condition:
            run_eval()

    run_eval()


if __name__ == '__main__':
    tf.app.run()
