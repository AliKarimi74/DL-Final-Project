import logging
import os
from config import config


class Logger:

    def __init__(self):
        self.name = ''
        self.out = None
        self.second_out = None

    def set_model_name(self, name):
        self.name = name + '-'

    def file(self):
        if self.out is None:
            file_path = self.name + config.log_path
            secondary_path = os.path.join(config.secondary_path, file_path)
            self.out = open(file_path, 'a+')
            try:
                self.second_out = open(secondary_path, 'a+')
                print('Setup secondary log file complete.')
            except IOError:
                self.second_out = None
        return self.out, self.second_out

    def write(self, msg):
        for f in self.file():
            if f is not None:
                f.write(msg)

    def log(self, msg, add_tailing_row=False):
        msg = str(msg)
        if add_tailing_row:
            msg += '\n' + '-'*200
        logging.info(msg)
        self.write(msg + '\n')


logger = Logger()
log = logger.log
