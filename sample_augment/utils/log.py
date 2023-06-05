import logging


class ColorLogFormatter(logging.Formatter):
    """
        custom log formatter for color outputs based on log level.
        see https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(levelname)s: %(message)s'  # our simplified format
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, level=logging.INFO):
    module_logger = logging.getLogger(name)
    module_logger.setLevel(level)

    # create console handler and set level that should be printed
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = ColorLogFormatter()
    handler.setFormatter(formatter)
    module_logger.addHandler(handler)

    return module_logger


log = get_logger('main logger')
