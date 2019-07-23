import tensorflow as tf
import numpy as np

from configuration import config, h_params
from utils.attention import BahdanauAttention


class RNNCell(tf.keras.layers.Layer):
    def __init__(self, units, memory):
        super(RNNCell, self).__init__()
        self.units = units
        self.cell = tf.keras.layers.GRUCell(self.units)
        # memory shape = (batch_size, n_locations, dim)
        self.memory = memory
        self.attention = BahdanauAttention(h_params.attention_units)
        self.out_projection = tf.keras.layers.Dense(self.units, activation='tanh', use_bias=False)

    def build(self, input_shape):
        input_shape = (input_shape[0], input_shape[1] + self.units)
        self.cell.build(input_shape)
        self.built = True

    def call(self, inputs, states):
        # inputs shape = (batch_size, embedding_dim)

        # cell_state shape = (batch_size, rnn_dim)
        # o_before shape = (batch_size, rnn_dim)
        cell_state, o_before = states

        # cell_input shape = (batch_size, embedding_dim + rnn_dim)
        cell_input = tf.concat([inputs, o_before], axis=-1)

        # output shape = (batch_size, rnn_dim)
        # new_state shape = (batch_size, rnn_dim)
        output, new_state = self.cell.call(inputs=cell_input, states=[cell_state])
        new_state = new_state[0]

        # context shape = (batch_size, context_dim)
        context = self._compute_context_vector(cell_state)

        # project_input shape = (batch_size, context_dim + rnn_dim)
        project_input = tf.concat([context, new_state], axis=-1)

        # output shape = (batch_size, rnn_dim)
        output = self.out_projection(project_input)

        return output, [new_state, output]

    def _compute_context_vector(self, hidden_state):
        # memory shape = (batch_size, n_locations, dim)

        # without attention
        if not h_params.use_attention:
            memory_shape = self.memory.get_shape().as_list()
            if h_params.flatten_image_features:
                # flat all locations features
                dims = memory_shape[1] * memory_shape[2]
                context_vector = tf.reshape(self.memory, [-1, dims])  # (batch_size, n_location * dim)
            else:
                # mean over feature locations
                context_vector = tf.reduce_mean(self.memory, axis=1)  # (batch_size, dim)

        # with attention
        else:
            context_vector, attention_weights = self.attention(self.memory, hidden_state)

        return context_vector

    @property
    def state_size(self):
        return [self.cell.state_size] + [self.units]

    @property
    def output_size(self):
        return self.units

    def _get_zero_output_state(self, inputs, batch_size, dtype):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        output_zero_state = tf.zeros([batch_size, self.units], dtype=dtype)
        return output_zero_state

    def get_init_state(self, encoder_state, batch_size, dtype):
        output_zero_state = self._get_zero_output_state(None, batch_size, dtype)
        return [encoder_state, output_zero_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        output_zero_state = self._get_zero_output_state(inputs, batch_size, dtype)
        return [self.cell.get_initial_state(inputs, batch_size, dtype)] + [output_zero_state]


class CustomRNN(tf.keras.Model):

    def __init__(self, start_token, features):
        super(CustomRNN, self).__init__()
        self.start_token = start_token
        self.embedding = tf.keras.layers.Embedding(config.vocab_size, h_params.embedding_dim)
        self.cell = RNNCell(h_params.decoder_rnn_dim, features)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True, return_state=True)
        self.projection = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

    def call(self, formula, encoder_state):
        sampling = formula is None
        batch_size = tf.shape(encoder_state)[0]

        init_state = self.cell.get_init_state(encoder_state, batch_size, dtype=tf.float32)

        # teacher forcing
        if not sampling:
            cell_input = self.embedding(formula)                        # (batch_size, n_times, embedding_dim)
            output = self.rnn(cell_input, initial_state=init_state)[0]  # (batch_size, n_times, rnn_dim)
            logits = self.projection(output)                            # (batch_size, n_times, vocab_size)
            return logits, None, None

        # generating
        else:
            n_times = config.max_generate_steps - 1
            start_emb = np.array([self.start_token])
            cell_input = tf.tile(start_emb, multiples=[batch_size])     # (batch_size, )
            state = init_state

            predictions, probs = [], []
            for i in range(n_times):
                cell_input = self.embedding(cell_input)                 # (batch_size, embedding_dim)
                # output shape = (batch_size, rnn_dim)
                output, state = self.cell(inputs=cell_input, states=state)
                logits = self.projection(output)                        # (batch_size, vocab_size)
                cell_input = tf.reshape(tf.random.categorical(logits, 1, dtype=tf.int32), [-1])  # (batch_size,)
                index = tf.concat([tf.expand_dims(tf.range(0, batch_size), axis=1),
                                   tf.expand_dims(cell_input, axis=1)], axis=-1)
                p = tf.nn.softmax(logits)
                p = tf.gather_nd(p, index)
                predictions.append(tf.expand_dims(cell_input, axis=1))
                probs.append(tf.expand_dims(p, axis=1))

            # preds shape = (batch_size, n_times)
            predictions = tf.concat(predictions, axis=1)
            probs = tf.concat(probs, axis=1)
            pp = self.compute_pp(probs)
            return None, predictions, pp

    def compute_pp(self, probs):
        n = tf.shape(probs)[1]
        p = 1 / probs
        d = tf.cast(1/n, dtype=tf.float32)
        p = p ** d
        p = tf.reduce_prod(p, axis=1)
        pp = tf.reduce_mean(p)
        return pp
