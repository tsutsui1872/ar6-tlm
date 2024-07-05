import logging

def get_logger(name):
    """Get a custom logger

    Parameters
    ----------
    name
        Logger name

    Returns
    -------
        Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if len(logger.handlers) == 0:
        # create console handler
        handler = logging.StreamHandler()
        handler.setLevel('INFO')

        # create formatter
        formatter = logging.Formatter(
            '[%(asctime)s %(name)s] %(levelname)s:%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)

        # add handler to logger
        logger.addHandler(handler)

    return logger


class MyExecError(Exception):
    pass


logger = get_logger(__name__)