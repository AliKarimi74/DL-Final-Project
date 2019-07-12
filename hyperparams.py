import tensorflow as tf
from collections import namedtuple

"""
Some information about hyper-parameters:
- `flatten_image_features`: 
    this option only used in models without attention. True means all image features
    flatted and concat with input embedding. This can hugely increase number of decoder parameters.
    While False mean one operation reduce different locations features by mean. 
"""

HyperParams = namedtuple('HyperParams',
                         'batch_size learning_rate '
                         'cnn_first_layer_filters add_positional_embed '
                         'row_encoder_rnn_dim '
                         'embedding_dim decoder_rnn_dim use_attention flatten_image_features')

h_params = HyperParams(
    batch_size=64,
    learning_rate=1e-3,
    cnn_first_layer_filters=32,
    add_positional_embed=False,
    row_encoder_rnn_dim=None,
    embedding_dim=64,
    decoder_rnn_dim=256,
    use_attention=False,
    flatten_image_features=False,
)
