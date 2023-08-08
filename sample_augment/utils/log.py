import io
import logging
import pprint
import sys


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
    script_location = " (%(filename)s:%(lineno)d)"  # add script location for warnings/errors
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + script_location + reset,
        logging.ERROR: red + format + script_location + reset,
        logging.CRITICAL: bold_red + format + script_location + reset
    }

    def format(self, record):
        # Pad the level name with spaces to a length of 7 (length of 'WARNING')
        record.levelname = f"{record.levelname:<7}"

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, level):
    # If name is the root logger, clear its handlers
    # if name == 'root' or name == '':
    #     root_logger = logging.getLogger()
    #     for handler in root_logger.handlers[:]:
    #         root_logger.removeHandler(handler)

    module_logger = logging.getLogger(name)
    module_logger.setLevel(level)
    module_logger.propagate = False  # Prevent this logger from passing messages to its parent

    if not module_logger.handlers:  # Only add handler if the logger has no handlers
        # create console handler and set level that should be printed
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = ColorLogFormatter()
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)

    return module_logger


log_level = logging.INFO
log = get_logger('sampling-aug', log_level)


# other utillity method related to printing / logging
class ShorteningPrettyPrinter(pprint.PrettyPrinter):
    def _format(self, obj, stream, indent, allowance, context, level):
        if isinstance(obj, (list, tuple)) and len(obj) > 10:
            # Customize this to change the threshold or the message
            stream.write(f"(length {len(obj)}...) {type(obj).__name__} with first 5 items: {obj[:5]} ...")
        else:
            # noinspection PyUnresolvedReferences,PyProtectedMember
            super()._format(obj, stream, indent, allowance, context, level)


def short_pformat(data):
    printer = ShorteningPrettyPrinter()
    return printer.pformat(data)


class SuppressSpecificPrint:
    def __init__(self, suppressed_message):
        self.suppressed_message = suppressed_message

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self._captured_stdout = io.StringIO()
        sys.stderr = self._captured_stderr = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        stdout_val = self._captured_stdout.getvalue()
        if self.suppressed_message not in stdout_val:
            self._original_stdout.write(stdout_val)

        stderr_val = self._captured_stderr.getvalue()
        if self.suppressed_message not in stderr_val:
            self._original_stderr.write(stderr_val)

        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
