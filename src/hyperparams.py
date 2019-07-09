from collections import namedtuple

HyperParams = namedtuple('HyperParams',
                         'batch_size embedding_dim')

h_params = HyperParams(
    batch_size=32,
    embedding_dim=64
)
