from collections import namedtuple

"""
Some information about hyper-parameters:
- `flatten_image_features`: 
    this option only used in models without attention. True means all image features
    flatted and concat with input embedding. This can hugely increase number of decoder parameters.
    While False mean one operation reduce different locations features by mean. 
"""

HyperParams = namedtuple('HyperParams',
                         'batch_size '
                         'learning_rate learning_decay_rate learning_decay_step '
                         'cnn_first_layer_filters add_positional_embed '
                         'row_encoder_rnn_dim '
                         'embedding_dim decoder_rnn_dim use_attention flatten_image_features')

row_encoder_dim = 256

h_params = HyperParams(
    batch_size=32,
    learning_rate=1e-3,
    learning_decay_rate=0.97,
    learning_decay_step=1000,
    cnn_first_layer_filters=64,
    add_positional_embed=False,
    row_encoder_rnn_dim=row_encoder_dim,
    embedding_dim=80,
    decoder_rnn_dim=2 * row_encoder_dim,
    use_attention=True,
    flatten_image_features=True,
)
