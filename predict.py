import numpy as np

from init import *
from data.data_preprocess import read_single_image

flags.DEFINE_string('image_path', 'img.png', 'Path to formula image')
FLAGS = flags.FLAGS


def main(args):
    def log(msg, new_section=False):
        LOG(msg, new_section)

    path = FLAGS.image_path
    path = path.strip('\'\"')

    # read image
    if os.path.isfile(path):
        image = read_single_image(path)
        image = np.expand_dims(image, 0)
    else:
        log('There is no image at: {}'.format(path))
        return

    model, sess, generator, _, _, _ = initialize('train', is_evaluation=True)

    prediction = model.predict(sess=sess, images=image)[0]
    prediction = generator.decode_formulas(prediction)[0]

    log('Model prediction:')
    log(prediction)


if __name__ == '__main__':
    tf.app.run()
