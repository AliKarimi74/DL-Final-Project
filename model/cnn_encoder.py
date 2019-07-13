import tensorflow as tf

from configuration import config, h_params
from utils.t2t_positional_embedding import add_timing_signal_nd


class CNNEncoder(tf.keras.Model):

    def __init__(self):
        super(CNNEncoder, self).__init__()

        first_down_sample_rate = 2 if h_params.use_attention else 3
        first_second_pool = (first_down_sample_rate, first_down_sample_rate)
        third_layer_pool = None if h_params.use_attention or not h_params.flatten_image_features else (2, 2)

        layers_filter_coefficient = [1, 2, 4, 4, 8, 8]
        layers_kernel_size = [3, 3, 3, 3, 3, 3]
        layers_pooling_size = [first_second_pool, first_second_pool, third_layer_pool, (2, 1), (1, 2), None]
        layers_pooling_stride = [first_second_pool, first_second_pool, third_layer_pool, (2, 1), (1, 2), None]

        self.conv_layers = []
        self.pool_layers = []
        with tf.variable_scope('cnn_encoder', reuse=tf.AUTO_REUSE):
            for i in range(len(layers_filter_coefficient)):
                filters = h_params.cnn_first_layer_filters * layers_filter_coefficient[i]
                kernel_size = layers_kernel_size[i]
                pad = 'same'
                act = tf.nn.relu
                pool_size = layers_pooling_size[i]
                pool_stride = layers_pooling_stride[i]

                conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                                    padding=pad, activation=act)
                self.conv_layers.append(conv_layer)

                pool_layer = None
                if pool_size is not None:
                    pool_layer = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_stride, padding=pad)
                self.pool_layers.append(pool_layer)

    def call(self, images, **kwargs):
        images = tf.cast(images, tf.float32) / 255.
        images = tf.expand_dims(images, axis=-1)
        # images shape (n, 60, 400, 1)

        with tf.variable_scope('cnn_encoder', reuse=tf.AUTO_REUSE):
            last_out = images
            for i in range(len(self.conv_layers)):
                out = self.conv_layers[i](last_out)
                if self.pool_layers[i] is not None:
                    out = self.pool_layers[i](out)
                last_out = out

            # add positional embedding to tensor
            if h_params.add_positional_embed:
                last_out = add_timing_signal_nd(last_out)

            return last_out
