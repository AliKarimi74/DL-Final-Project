import logging
import os
from config import config


class Logger:

    def __init__(self):
        self.name = ''
        self.logger = None

    def set_model_name(self, name):
        self.name = name + '-'

    def get_logger(self):
        if self.logger is None:
            root_logger = logging.getLogger()
            log_format = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
            root_logger.setLevel(logging.INFO)

            file_path = self.name + config.log_path
            secondary_path = os.path.join(config.secondary_path, file_path)

            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(log_format)
            root_logger.addHandler(file_handler)

            try:
                secondary_out = logging.FileHandler(secondary_path)
                secondary_out.setFormatter(log_format)
                root_logger.addHandler(secondary_out)
                root_logger.info('Setup secondary log file complete.')
            except IOError:
                root_logger.info('Can\'t setup secondary log file.')

            self.logger = root_logger
        return self.logger

    def write(self, msg):
        self.get_logger().info(msg)

    def log(self, msg, add_tailing_row=False):
        msg = str(msg)
        if add_tailing_row:
            msg += '\n' + '-'*200

        self.write(msg)


logger = Logger()
log = logger.log
