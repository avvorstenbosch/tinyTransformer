import logging

LOGGING_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def setup_logger(name, level="INFO"):
    """
    Set up a logger that prints logs to both the console and a file.

    Parameters
    ----------
    name : str,
        The name of the logger, this should be set to __name__.
    level : str, optional
        The logging level to use (default is 'INFO'). The available levels are
        'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', and 'NOTSET'.

    Returns
    -------
    logging.Logger
        The logger instance that was created.

    Examples
    --------
    >>> from logger import setup_logger
    >>> logger = setup_logger(level='DEBUG')
    >>> logger.debug('This is a debug message')
    """
    # Set up the logger with a format that includes the filename and date
    logger = logging.getLogger(name)
    level = LOGGING_LEVELS[level]
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a handler that prints logs to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a handler that prints logs to a file with the current filename and date
    log_filename = f"{name}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
