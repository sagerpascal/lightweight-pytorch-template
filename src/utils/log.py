import datetime
import logging
import sys
from pathlib import Path


def setup_logging():
    """ Setup the logger """
    logging.basicConfig(filename=Path('logs') / '{}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")),
                        filemode="w", level=logging.DEBUG)

    logger = logging.getLogger()  # the root logger
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


def format_logs(logs):
    """ format logs in tqdm """
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s