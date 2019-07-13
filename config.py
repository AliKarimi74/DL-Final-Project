from collections import namedtuple

Config = namedtuple('Config', 'dataset_path '
                              'vocab_path vocab_size '
                              'save_path log_path secondary_path '
                              'n_epochs '
                              'max_generate_steps '
                              'log_every eval_every')

config = Config(
    dataset_path='Dataset',
    vocab_path='res/vocab.json',
    vocab_size=400,
    save_path='runs',
    log_path='log.txt',
    secondary_path='/content/gdrive/My Drive/Deep learning/Project',
    n_epochs=12,
    max_generate_steps=5,
    log_every=10,
    eval_every=1000
)
