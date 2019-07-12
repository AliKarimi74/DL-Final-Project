import tensorflow as tf

from hyperparams import h_params


class RowEncoder(tf.keras.Model):

    def __init__(self):
        super(RowEncoder, self).__init__()

        with tf.variable_scope('row_encoder', reuse=tf.AUTO_REUSE):
            gpu_available = tf.test.is_gpu_available()
            cell = tf.keras.layers.CuDNNGRU if gpu_available else tf.keras.layers.GRU
            self.units = h_params.row_encoder_rnn_dim
            if self.units is not None:
                self.fw_rnn = cell(units=self.units,
                                   return_sequences=True,
                                   return_state=True)
                self.bw_rnn = cell(units=self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   go_backwards=True)
                self.rnn_layer = tf.keras.layers.Bidirectional(layer=self.fw_rnn, backward_layer=self.bw_rnn,
                                                               name='bi_lstm')

    def call(self, encoded_images, **kwargs):
        with tf.variable_scope('row_encoder', reuse=tf.AUTO_REUSE):
            shape = encoded_images.get_shape().as_list()
            n_rows, n_cols, n_channels = shape[1:4]
            image_rows = tf.reshape(encoded_images, shape=[-1, n_cols, n_channels])

            final_channels = n_channels
            out = image_rows
            state = None
            if self.units is not None:
                final_channels = 2 * h_params.row_encoder_rnn_dim
                rnn_out = self.rnn_layer(out)
                out, state = rnn_out[0], tf.concat(rnn_out[1:], axis=-1)
                state = tf.reshape(state, shape=[tf.shape(encoded_images)[0], -1, final_channels])
                state = tf.reduce_mean(state, axis=1)
            out = tf.reshape(out, shape=[-1, n_rows * n_cols, final_channels])

            return out, state

