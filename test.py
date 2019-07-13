from init import *
from eval.evaluation import evaluation
from utils.logger import log as LOG

flags.DEFINE_string('dataset', 'test', 'Dataset to evaluate on it. (validation or train)')
flags.DEFINE_string('save_prediction_dir', '.', 'Path of directory for saving prediction')
FLAGS = flags.FLAGS


def main(args):
    def log(msg, new_section=False):
        LOG(msg, new_section)

    mode = 'validation'
    if FLAGS.dataset == 'test' or FLAGS.dataset == 'train':
        mode = FLAGS.dataset
    model, sess, train_set, saver, _, _ = initialize(mode, log_suffix='-test')
    save_path = None
    if len(FLAGS.save_prediction_dir) > 0:
        save_file_name = 'predicted_{}_formulas.txt'.format(mode)
        save_path = os.path.join(FLAGS.save_prediction_dir, save_file_name)

    small_data = FLAGS.check_on_small_data
    n_epochs, per_limit = 1, 0.01 if small_data else None

    log('Start testing ' + ('on small data' if small_data else '...'))
    evaluation(session=sess, model=model, mode=mode, percent_limit=per_limit, save_path=save_path)


if __name__ == '__main__':
    tf.app.run()
