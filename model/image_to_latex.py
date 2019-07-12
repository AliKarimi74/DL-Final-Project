import tensorflow as tf

from hyperparams import h_params
from config import config
from utils.logger import log as LOG
from .cnn_encoder import CNNEncoder
from .row_encoder import RowEncoder
from .decoders import Decoder


class ImageToLatexModel(object):

    def __init__(self, start_token, pad_token):
        self.start_token = start_token
        self.pad_token = pad_token

        self.__build_graph()
        self.loss, self.acc = self.__loss()
        self.train_op = self.__optimization()

        self.prediction = self.__predict()

        self.__print_summary()

    def __build_graph(self):
        self.__place_holders()

        # encoder
        self.encoder = CNNEncoder()
        self.encode_image = self.encoder(self.images)                           # (batch_size, n_rows, n_cols, filters)
        self.first_cnn_filters = self.encoder.conv_layers[0].weights[0]
        self.row_encoder = RowEncoder()
        encode_image, state = self.row_encoder(self.encode_image)               # (batch_size, n_rows, n_cols, filters)

        # decoder
        self.decoder = Decoder(self.start_token)
        self.logits = self.decoder(encode_image, self.formulas[:, :-1], state)  # (batch_size, n_times, vocab_size)

    def __place_holders(self):
        with tf.variable_scope('place_holders', reuse=tf.AUTO_REUSE):
            self.images = tf.placeholder(tf.uint8, shape=[None, 60, 400], name='images')
            self.formulas = tf.placeholder(tf.int32, shape=[None, config.max_generate_steps], name='formulas')

    def __loss(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

            real = self.formulas[:, 1:]
            pred = self.logits

            ce_loss = loss_obj(y_true=real, y_pred=pred)                        # (batch_size, n_times)
            mask = tf.logical_not(tf.equal(real, self.pad_token))
            mask = tf.cast(mask, dtype=ce_loss.dtype)

            ce_loss *= mask
            # p = tf.argmax(pred, axis=2)
            # acc = tf.metrics.accuracy(labels=real, predictions=p, weights=mask)[0]

            loss = tf.reduce_mean(ce_loss)
            return loss, None

    def __optimization(self):
        with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):
            self.step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(h_params.learning_rate, self.step,
                                                            decay_steps=300, decay_rate=0.9)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = self.optimizer.minimize(self.loss, global_step=self.step)
            return train_op

    def __predict(self):
        with tf.variable_scope('predict', reuse=tf.AUTO_REUSE):
            encode_image, state = self.row_encoder(self.encoder(self.images))
            return self.decoder(encode_image, None, state)

    def __feed_dict(self, images, formulas):
        return {
            self.images: images,
            self.formulas: formulas
        }

    def __print_summary(self):
        encoder_params = self.encoder.count_params()
        row_encoder_params = self.row_encoder.count_params()
        decoder_params = self.decoder.count_params()

        template = ' | {0:<15} | {1:<15} |'
        horizontal_row = template.format('-'*15, '-'*15)

        def log(comp, params, add_trailing_row=False, add_heading_row=False):
            if add_heading_row:
                LOG(horizontal_row)
            LOG(template.format(comp, str(params)))
            if add_trailing_row:
                LOG(horizontal_row)

        log('Component', 'Parameters', True, True)
        log('Encoder', encoder_params)
        log('Row Encoder', row_encoder_params)
        log('Decoder', decoder_params, True)
        log('Sum', encoder_params + row_encoder_params + decoder_params, True)

    def train_step(self, sess, images, formulas):
        _, loss, step, lr = \
            sess.run([self.train_op, self.loss, self.step, self.learning_rate],
                     feed_dict=self.__feed_dict(images, formulas))
        return loss, step, lr

    def predict(self, sess, images):
        dic = {self.images: images}
        predictions = sess.run([self.prediction], feed_dict=dic)
        return predictions
