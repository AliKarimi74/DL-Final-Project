from collections import namedtuple

HyperParams = namedtuple('HyperParams',
                         'batch_size learning_rate '
                         'cnn_first_layer_filters add_positional_embed '
                         'row_encoder_rnn_dim '
                         'embedding_dim decoder_rnn_dim use_attention attention_units')

h_params = HyperParams(
    batch_size=64,
    learning_rate=1e-3,
    cnn_first_layer_filters=64,
    add_positional_embed=False,
    row_encoder_rnn_dim=128,
    embedding_dim=64,
    decoder_rnn_dim=256,            # if attention used decoder hidden state dim MUST be equal to 2 * row encoder dim
    use_attention=False,
    attention_units=128
)
