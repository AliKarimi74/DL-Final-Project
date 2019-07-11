import tensorflow as tf
import numpy as np

from config import config
from hyperparams import h_params


class BaseDecoder(tf.keras.Model):

    def __init__(self, start_token):
        super(BaseDecoder, self).__init__()

        self.start_token = start_token
        with tf.variable_scope('base_decoder', reuse=tf.AUTO_REUSE):
            self.embedding = tf.keras.layers.Embedding(config.vocab_size, h_params.embedding_dim)
            self.rnn = tf.keras.layers.LSTM(units=h_params.decoder_rnn_dim,
                                            return_sequences=True,
                                            return_state=True)

            # projection
            self.projection_fc = tf.keras.layers.Dense(config.vocab_size)

    def call(self, image_encoding, formula, **kwargs):
        with tf.variable_scope('base_decoder', reuse=tf.AUTO_REUSE):
            images_shape = image_encoding.get_shape().as_list()
            features_shape = [-1, images_shape[1] * images_shape[2], images_shape[3]]
            features = tf.reshape(image_encoding, shape=features_shape)     # (batch_size, n_location, dim)

            if h_params.flatten_image_features:
                # flat all locations features
                dims = features_shape[1] * features_shape[2]
                features = tf.reshape(features, [-1, dims])                 # (batch_size, n_location * dim)
            else:
                # mean over feature locations
                features = tf.reduce_mean(features, axis=1)                 # (batch_size, dim)

            if formula is None:
                return self._decode_with_sampling(features)
            else:
                return self._teacher_forcing_decode(features, formula)

    def _teacher_forcing_decode(self, features, formula):
        x = self.embedding(formula)                                         # (batch_size, T, embedding_dim)

        features = tf.expand_dims(features, axis=1)                         # (batch_size, 1, dim)
        n_times = tf.shape(x)[1]
        features = tf.tile(features, multiples=[1, n_times, 1])             # (batch_size, T, dim)
        rnn_input = tf.concat([x, features], axis=-1)                       # (batch_size, T, embedding_dim + dim)

        rnn_out = self.rnn(rnn_input)[0]                                    # (batch_size, T, rnn_dim)
        logits = self.projection_fc(rnn_out)                                # (batch_size, T, vocab_size)

        return logits

    def _decode_with_sampling(self, features):
        def augment(inputs):
            return tf.concat([inputs, features], axis=-1)                   # (batch_size, embedding_dim + dim)

        batch_size = tf.shape(features)[0]
        start_emb = self.embedding(np.array([self.start_token]))
        start_emb = tf.tile(start_emb, multiples=[batch_size, 1])           # (batch_size, embedding_dim)

        state = self.rnn.get_initial_state(start_emb)
        inp = start_emb
        output = []
        cell = self.rnn.cell
        for t in range(config.max_generate_steps):
            out, state = cell(augment(inp), state)
            logits = self.projection_fc(out)                                # (batch_size, vocab_size)
            samples = tf.reshape(tf.random.categorical(logits, 1), [-1])    # (batch_size)

            output += [samples]
            inp = self.embedding(samples)                                   # (batch_size, embedding_dim)

        output = tf.transpose(tf.convert_to_tensor(output))                 # (batch_size, max_generate_steps)
        return output
