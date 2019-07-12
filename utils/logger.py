import logging
from config import config


class Logger:

    def __init__(self):
        self.name = ''
        self.out = None

    def set_model_name(self, name):
        self.name = name + '-'

    def file(self):
        if self.out is None:
            file_path = self.name + config.log_path
            self.out = open(file_path, 'a+')
        return self.out

    def log(self, msg, add_tailing_row=False):
        msg = str(msg)
        if add_tailing_row:
            msg += '\n' + '-'*200
        logging.info(msg)
        self.file().write(msg + '\n')


logger = Logger()
log = logger.log
