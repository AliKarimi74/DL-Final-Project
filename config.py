from collections import namedtuple

Config = namedtuple('Config', 'dataset_path vocab_path vocab_size save_path '
                              'n_epochs '
                              'max_generate_steps '
                              'log_every eval_every_epoch')

config = Config(
    dataset_path='Dataset',
    vocab_path='res/vocab.json',
    vocab_size=500,
    save_path='runs',
    n_epochs=10,
    max_generate_steps=400,
    log_every=40,
    eval_every_epoch=1
)
