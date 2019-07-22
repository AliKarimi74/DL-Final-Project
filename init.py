import os

import tensorflow as tf

from configuration import config, h_params
from data.data_generator import DataGenerator
from model.image_to_latex import ImageToLatexModel
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


def initialize(mode='train', is_evaluation=False):
    def log(msg, trailing_line=False):
        LOG(msg, trailing_line)

    dataset = DataGenerator(mode)
    start_token, pad_token = dataset.data.start_token(), dataset.data.pad_token()

    model_name = FLAGS.model_name
    model_path = os.path.join(config.save_path, model_name + '.ckpt')
    secondary_model_path = os.path.join(config.secondary_path, config.save_path, model_name + '.ckpt')
    logger.set_model_name(model_name + ('-test' if is_evaluation else ''))

    log(config, True)
    log(h_params, True)

    gpu_is_available = tf.test.is_gpu_available()
    log('GPU is available: ' + str(gpu_is_available))
    log('Start building graph')
    tf.reset_default_graph()
    model = ImageToLatexModel(start_token, pad_token)
    log('Graph building finished!', True)

    # params = tf.trainable_variables()
    # print('Trainable params:')
    # for p in params:
    #     print(p)

    sess = tf.Session()
    init_opt = tf.global_variables_initializer()

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    restored = False
    if FLAGS.load_from_previous or is_evaluation:
        def load(path, name):
            nonlocal restored, saver, sess
            if restored:
                return
            try:
                saver.restore(sess, path)
                step = sess.run([model.step])[0]
                log('Model restored from last session in {}, current step: {}'.format(name, step))
                # run_eval()
                restored = True
            except Exception as e:
                log('Can\'t load from previous session in {}, error: {}'.format(name, e))

        load(model_path, 'model path')
        load(secondary_model_path, 'secondary model path')
    if not restored:
        sess.run(init_opt)

    return model, sess, dataset, saver, model_path, secondary_model_path
