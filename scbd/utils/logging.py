import logging
import sys


def get_stdout_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.flush = sys.stdout.flush
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    return logger