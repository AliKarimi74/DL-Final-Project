from collections import namedtuple

Config = namedtuple('Config', 'dataset_path vocab_path vocab_size save_path log_path '
                              'n_epochs '
                              'max_generate_steps '
                              'log_every eval_every_epoch')

config = Config(
    dataset_path='Dataset',
    vocab_path='res/vocab.json',
    vocab_size=400,
    save_path='runs',
    log_path='log.txt',
    n_epochs=20,
    max_generate_steps=20,
    log_every=10,
    eval_every_epoch=1
)
