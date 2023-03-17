import logging


def get_logger(log_file='test.log'):
    logger = logging.getLogger()

    fh = logging.FileHandler(log_file)  # file log
    ch = logging.StreamHandler()  # console log
    # format
    fm = logging.Formatter(fmt='%(asctime)s %(message)s',
                           datefmt='[%Y-%m-%d %H:%M:%S]')

    fh.setFormatter(fm)
    ch.setFormatter(fm)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel('INFO')  # set level

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info('message')
