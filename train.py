import tensorflow as tf
import numpy as np

from init import *
from configuration import config
from eval.evaluation import evaluation
from utils.logger import log as LOG


def main(args):
    def log(msg, new_section=False):
        LOG(msg, new_section)

    model, sess, train_set, saver, model_path, secondary_model_path = initialize('train')

    gpu_is_available = tf.test.is_gpu_available()
    mini_loss_history, loss_hist = [], []
    small_data = FLAGS.check_on_small_data
    n_epochs, per_limit = (500, 0.01) if small_data else (config.n_epochs, None)
    validation_set = 'train' if small_data else 'validation'

    log_every = config.log_every if gpu_is_available else 2
    eval_every_epoch = config.eval_every if not small_data else n_epochs // 20
    log_template = 'Epoch {0} ({1}), step = {2} => Loss: {3:1.3f}, lr: {4}'  # , Accuracy: {4:2.2f}'

    def run_eval():
        nonlocal sess, model, saver, validation_set, per_limit, model_path, secondary_model_path
        if not small_data:
            path = saver.save(sess, model_path)
            log('Model saved in {}'.format(path))
            try:
                saver.save(sess, secondary_model_path)
                log('Model saved in {}'.format(secondary_model_path), new_section=True)
            except Exception as e:
                log('Can\' save in {}, error: {}'.format(secondary_model_path, e), new_section=True)
        evaluation(session=sess, model=model, mode=validation_set, percent_limit=per_limit)

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

        if step % eval_every_epoch == 0:
            run_eval()

    run_eval()


if __name__ == '__main__':
    tf.app.run()
