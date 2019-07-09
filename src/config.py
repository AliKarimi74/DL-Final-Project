from collections import namedtuple

Config = namedtuple('Config', 'dataset_path max_number_of_tokens n_epochs')

config = Config(
    dataset_path='../Dataset',
    max_number_of_tokens=500,
    n_epochs=1
)
