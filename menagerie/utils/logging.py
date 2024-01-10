"""Logging utilities."""

import logging

from loguru import logger


def create_logger():
    # Create a logger
    logger = logging.getLogger("menagerie")
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define the format for the log messages
    formatter = logging.Formatter(
        "%(levelname)s - (%(filename)s:%(lineno)d) - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


# def create_logger() -> "loguru.Logger":
#     """Create a logger."""
#     # logger = logging.getLogger(__name__)
#     # logger.setLevel(logging.INFO)
#     # handler = logging.StreamHandler()
#     # formatter = logging.Formatter(
#     #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     # )
#     # handler.setFormatter(formatter)
#     # logger.addHandler(handler)

#     # Configure standard logging to forward to Loguru
#     logging.basicConfig(handlers=[LoguruHandler()], level=logging.INFO)

#     # PyTorch Lightning specific configuration
#     lightning_logger = logging.getLogger("pytorch_lightning")
#     lightning_logger.setLevel(logging.INFO)  # or any level you prefer

#     # Now you can use Loguru for logging
#     logger.info("This message goes to Loguru")

#     return logger


class LoguruHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
