import tensorflow as tf

from hyperparams import h_params


class RowEncoder(tf.keras.Model):

    def __init__(self):
        super(RowEncoder, self).__init__()

        with tf.variable_scope('row_encoder', reuse=tf.AUTO_REUSE):
            gpu_available = tf.test.is_gpu_available()
            cell = tf.keras.layers.CuDNNLSTM if gpu_available else tf.keras.layers.LSTM
            self.units = h_params.row_encoder_rnn_dim
            if self.units is not None:
                self.fw_rnn = cell(units=self.units,
                                   return_sequences=True,
                                   return_state=False)
                self.bw_rnn = cell(units=self.units,
                                   return_sequences=True,
                                   return_state=False,
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
            if self.units is not None:
                final_channels = 2 * h_params.row_encoder_rnn_dim
                out = self.rnn_layer(out)
            out = tf.reshape(out, shape=[-1, n_rows * n_cols, final_channels])

            return out

