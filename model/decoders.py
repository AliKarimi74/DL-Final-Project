import tensorflow as tf
import numpy as np

from configuration import config, h_params
from utils.attention import BahdanauAttention


class Decoder(tf.keras.Model):

    def __init__(self, start_token):
        super(Decoder, self).__init__()

        self.start_token = start_token
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.embedding = tf.keras.layers.Embedding(config.vocab_size, h_params.embedding_dim)
            self.cell = tf.keras.layers.GRU(h_params.decoder_rnn_dim,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
            self.attention = BahdanauAttention(h_params.decoder_rnn_dim)

            # projection
            self.out_projection = tf.keras.layers.Dense(h_params.decoder_rnn_dim, activation='tanh', use_bias=False)
            self.projection = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

    def call(self, features, formula, init_state=None):
        with tf.variable_scope('decoder_call', reuse=tf.AUTO_REUSE):
            sampling = formula is None
            batch_size = tf.shape(features)[0]
            n_times = config.max_generate_steps if sampling else formula.get_shape().as_list()[1]
            start_emb = np.array([[self.start_token]])
            inp = tf.tile(start_emb, multiples=[batch_size, 1])                     # (batch_size, 1)

            state = init_state
            if state is None:
                state = self.cell.get_initial_state(features)
                if type(state) == list:
                    state = tf.concat(state, axis=-1)
            last_out_state = None
            outputs = []

            def loop_body_teacher_forcing(i):
                nonlocal outputs, formula, features, state, last_out_state
                inp = tf.expand_dims(formula[:, i], axis=1)                         # (batch_size, 1)
                logits, state, last_out_state = self.step(features, inp, state, last_out_state)
                outputs += [logits]

            def loop_body_sampling(i):
                nonlocal outputs, inp, features, state, last_out_state
                logits, state, last_out_state = self.step(features, inp, state, last_out_state)
                logits = tf.reshape(logits, shape=[-1, logits.shape[-1]])
                samples = tf.reshape(tf.random.categorical(logits, 1), [-1])        # (batch_size)
                samples = tf.expand_dims(samples, axis=1)
                outputs += [samples]
                inp = samples

            loop_body = loop_body_sampling if sampling else loop_body_teacher_forcing
            for i in range(n_times):
                loop_body(i)

            outputs = tf.concat(outputs, axis=1)
            return outputs

    def step(self, features, formula, hidden_state, last_out_state):
        # x shape = (batch_size, 1, emb_dim)
        x = self.embedding(formula)

        # last_out shape: (batch_size, 1, rnn_dim)
        if last_out_state is None:
            last_out_state = tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], h_params.decoder_rnn_dim])

        # input shape = (batch_size, 1, rnn_dim + emb_dim)
        inp = tf.concat([x, last_out_state], axis=-1)

        # cell_output shape = (batch_size, 1, rnn_dim)
        # state shape = (batch_size, rnn_dim)
        cell_output, state = self.cell(inp, initial_state=hidden_state)

        # context shape = (batch_size, features_dim)
        context_vector = self._compute_context_vector(features, state)

        # out_feed shape = (batch_size, 1, rnn_dim + features_dim)
        out_feed = tf.expand_dims(tf.concat([state, context_vector], axis=-1), axis=1)
        out_state = self.out_projection(out_feed)

        output = self.projection(out_state)
        # output shape = (batch_size, 1, vocab_size)

        return output, state, out_state

    def _compute_context_vector(self, features, hidden):
        # features shape = (batch_size, n_locations, dim)

        # without attention
        if not h_params.use_attention:
            features_shape = features.get_shape().as_list()
            if h_params.flatten_image_features:
                # flat all locations features
                dims = features_shape[1] * features_shape[2]
                context_vector = tf.reshape(features, [-1, dims])  # (batch_size, n_location * dim)
            else:
                # mean over feature locations
                context_vector = tf.reduce_mean(features, axis=1)  # (batch_size, dim)

        # with attention
        else:
            context_vector, attention_weights = self.attention(features, hidden)

        return context_vector
