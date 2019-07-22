import tensorflow as tf
import numpy as np

from configuration import config, h_params
from utils.logger import log as LOG
from .cnn_encoder import CNNEncoder
from .row_encoder import RowEncoder
from .decoders import CustomRNN


class ImageToLatexModel(object):

    def __init__(self, start_token, pad_token):
        self.start_token = start_token
        self.pad_token = pad_token

        self._build_graph()
        self.loss = self._loss()
        self.train_op = self._optimization()

        self.prediction = self._predict()

        self._print_summary()

    def _build_graph(self):
        self._place_holders()

        # encoder
        self.encoder = CNNEncoder()
        self.encode_image = self.encoder(self.images)                           # (batch_size, n_rows, n_cols, filters)
        self.first_cnn_filters = self.encoder.conv_layers[0].weights[0]
        self.row_encoder = RowEncoder()
        self.encode_image, self.encoder_state = \
            self.row_encoder(self.encode_image)                                 # (batch_size, n_rows, n_cols, filters)

        # decoder
        self.decoder = CustomRNN(self.start_token, self.encode_image)
        # logits shape = (batch_size, n_times-1, vocab_size)
        self.logits, _ = self.decoder(self.formulas[:, :-1], encoder_state=self.encoder_state)

    def _place_holders(self):
        with tf.variable_scope('place_holders', reuse=tf.AUTO_REUSE):
            self.images = tf.placeholder(tf.uint8, shape=[None, 60, 400], name='images')
            self.formulas = tf.placeholder_with_default(tf.zeros([1, config.max_generate_steps], dtype=tf.int32),
                                                        shape=[None, None], name='formulas')

    def _loss(self):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

            real = self.formulas[:, 1:]
            pred = self.logits

            ce_loss = loss_obj(y_true=real, y_pred=pred)                        # (batch_size, n_times)
            mask = tf.logical_not(tf.equal(real, self.pad_token))
            mask = tf.cast(mask, dtype=ce_loss.dtype)

            ce_loss *= mask

            loss = tf.reduce_mean(ce_loss)
            return loss

    def _optimization(self):
        with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):
            self.step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(h_params.learning_rate, self.step,
                                                            decay_steps=h_params.learning_decay_step,
                                                            decay_rate=h_params.learning_decay_rate)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = self.optimizer.minimize(self.loss, global_step=self.step)
            return train_op

    def _print_summary(self):
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

    def _predict(self):
        with tf.variable_scope('predict', reuse=tf.AUTO_REUSE):
            _, output = self.decoder(None, encoder_state=self.encoder_state)
            return output

    def train_batch(self, sess, images, formulas):
        _, loss, step, lr = \
            sess.run([self.train_op, self.loss, self.step, self.learning_rate],
                     feed_dict={self.images: images, self.formulas: formulas})
        return loss, step, lr

    def predict(self, sess, images):
        predictions = sess.run([self.prediction], feed_dict={self.images: images})
        return predictions
