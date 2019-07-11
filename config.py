from collections import namedtuple

Config = namedtuple('Config', 'dataset_path vocab_path vocab_size '
                              'n_epochs '
                              'max_generate_steps '
                              'eval_every')

config = Config(
    dataset_path='Dataset',
    vocab_path='res/vocab.json',
    vocab_size=500,
    n_epochs=1,
    max_generate_steps=400,
    eval_every=2000
)
